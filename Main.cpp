/************************************************************************
 * Copyright © 2020 The Multiphysics Modeling and Computation (M2C) Lab
 * <kevin.wgy@gmail.com> <kevinw3@vt.edu>
 ************************************************************************/

#include <time.h>
#include <petscdmda.h> //PETSc
#include <MeshGenerator.h>
#include <Output.h>
#include <VarFcnSG.h>
#include <VarFcnMG.h>
#include <VarFcnMGExt.h>
#include <VarFcnJWL.h>
#include <FluxFcnGodunov.h>
#include <SpaceInitializer.h>
#include <GradientCalculatorCentral.h>
#include <set>
#include <string>
using std::to_string;
#include <limits>

// for timing
//using std::chrono::high_resolution_clock;
//using std::chrono::duration_cast;
//using std::chrono::duration;
//using std::chrono::milliseconds;



int verbose;
double domain_diagonal;
double start_time; //start time in seconds
MPI_Comm m2c_comm;

int INACTIVE_MATERIAL_ID;

/*************************************
 * Main Function
 ************************************/
int main(int argc, char* argv[])
{
  
  //! Initialize MPI 
  //MPI_Init(NULL,NULL); //called together with all concurrent programs -> MPI_COMM_WORLD
  start_time = walltime(); //for timing purpose only (calls MPI_Wtime)

  //! Print header (global proc #0, assumed to be a M2C proc)
  //m2c_comm = MPI_COMM_WORLD; //temporary, just for the next few lines of code
  printHeader(argc, argv);

  //! Read user's input file (read the parameters)
  IoData iod(argc, argv);
  verbose = iod.output.verbose;

  //! Partition MPI, if there are concurrent programs
  MPI_Comm comm; //this is going to be the M2C communicator
  ConcurrentProgramsHandler concurrent(iod, MPI_COMM_WORLD, comm);
  m2c_comm = comm; //correct it
 
  //! Finalize IoData (read additional files and check for errors)
  iod.finalize();



  //! Initialize VarFcn (EOS, etc.) 
  std::set<int> vf_tracker;
  std::vector<VarFcnBase *> vf;
  for(int i=0; i<(int)iod.eqs.materials.dataMap.size(); i++)
    vf.push_back(NULL); //allocate space for the VarFcn pointers

  for(auto it = iod.eqs.materials.dataMap.begin(); it != iod.eqs.materials.dataMap.end(); it++) {
    int matid = it->first;
    vf_tracker.insert(matid);
    if(matid < 0 || matid >= (int)vf.size()) {
      print_error("*** Error: Detected error in the specification of material indices (id = %d).\n", matid);
      exit_mpi();
    }
    if(it->second->eos == MaterialModelData::STIFFENED_GAS)
      vf[matid] = new VarFcnSG(*it->second);
    else if(it->second->eos == MaterialModelData::MIE_GRUNEISEN)
      vf[matid] = new VarFcnMG(*it->second);
    else if(it->second->eos == MaterialModelData::EXTENDED_MIE_GRUNEISEN)
      vf[matid] = new VarFcnMGExt(*it->second);
    else if(it->second->eos == MaterialModelData::JWL)
      vf[matid] = new VarFcnJWL(*it->second);
    } else {
      print_error("*** Error: Unable to initialize variable functions (VarFcn) for the "
                  "specified material model.\n");
      exit_mpi();
    }
  }
  if(vf_tracker.size() != vf.size()) {
    print_error("*** Error: Detected error in the specification of material IDs.\n");
    exit_mpi();
  }
  vf_tracker.clear();   

  /*******************************************************/


  //! Initialize the exact Riemann problem solver.
  ExactRiemannSolverBase *riemann = ExactRiemannSolverData(vf, iod.exact_riemann);

  //! Initialize FluxFcn for the advector flux of the N-S equations
  FluxFcnBase *ff = FluxFcnGodunov(vf, iod);
  print_warning("*** Warning: Initialized flux calculator (FluxFcn) to the Godunov method.\n");

  //! Calculate mesh coordinates
  vector<double> xcoords, dx, ycoords, dy, zcoords, dz;
  MeshGenerator meshgen;
  meshgen.ComputeMeshCoordinatesAndDeltas(iod.mesh, xcoords, ycoords, zcoords, dx, dy, dz);
  domain_diagonal = sqrt(pow(iod.mesh.xmax - iod.mesh.x0, 2) +
                         pow(iod.mesh.ymax - iod.mesh.y0, 2) +
                         pow(iod.mesh.zmax - iod.mesh.z0, 2));
  
  //! Setup global mesh info
  GlobalMeshInfo global_mesh(xcoords, ycoords, zcoords, dx, dy, dz, false);

  //! Initialize PETSc -- TODO: check this..
  PETSC_COMM_WORLD = comm;
  PetscInitialize(&argc, &argv, argc>=3 ? argv[2] : (char*)0, (char*)0);

  //! Setup PETSc data array (da) structure for nodal variables
  DataManagers3D dms(comm, xcoords.size(), ycoords.size(), zcoords.size());

  //! Let global_mesh find subdomain boundaries and neighbors
  global_mesh.FindSubdomainInfo(comm, dms);

  //! Initialize space operator
  SpaceOperator spo(comm, dms, iod, vf, *ff, *riemann, global_mesh);

  //! Initialize interpolator
  InterpolatorBase *interp = new InterpolatorLinear(comm, dms, spo.GetMeshCoordinates(), spo.GetMeshDeltaXYZ());

  //! Initialize (sptial) gradient calculator
  GradientCalculatorBase *grad = new GradientCalculatorCentral(comm, dms, spo.GetMeshCoordinates(), spo.GetMeshDeltaXYZ(), *interp);
  
  /** Create a set that stores additional nodes/cells where solutions should *not* be updated
    * In the vast majority of cases, this set should be empty. Use it only when other options
    * are impossible or a lot more intrusive.*/
  std::set<Int3> spo_frozen_nodes;
  spo.SetPointerToFrozenNodes(&spo_frozen_nodes); 


  //! Allocate memory for V and ID 
  SpaceVariable3D V(comm, &(dms.ghosted1_5dof)); //!< primitive state variables
  SpaceVariable3D ID(comm, &(dms.ghosted1_1dof)); //!< material id

  //! Allocate memory for Phi. Initialize LevelSetOperators
  std::vector<LevelSetOperator*> lso;
  std::vector<SpaceVariable3D*>  Phi;
  std::vector<SpaceVariable3D*> NPhi; // unit normal, first derivative of Phi
  std::vector<SpaceVariable3D*> KappaPhi; // curvature, based on the second derivative of Phi

  std::set<int> ls_tracker;
  int ls_input_id_min = 9999, ls_input_id_max = -9999;
  for(auto it = iod.schemes.ls.dataMap.begin(); it != iod.schemes.ls.dataMap.end(); it++) {
    if(ls_tracker.find(it->first) != ls_tracker.end()){
      print_error("*** Error: Detected two level sets with the same id (%d).\n", it->first);
      exit_mpi();
    }
    ls_tracker.insert(it->first);
    ls_input_id_min = std::min(ls_input_id_min, it->first);
    ls_input_id_max = std::max(ls_input_id_max, it->first);
  } 
  if(ls_input_id_min<0 || ls_input_id_max>=(int)ls_tracker.size()){
    print_error("*** Error: Level set ids should start from 0 and have no gaps.\n"); 
    exit_mpi();
  }
  lso.resize(ls_tracker.size(),NULL);
  Phi.resize(ls_tracker.size(),NULL);
  NPhi.resize(ls_tracker.size(),NULL);
  KappaPhi.resize(ls_tracker.size(),NULL);
  ls_tracker.clear(); //for re-use
  for(auto it = iod.schemes.ls.dataMap.begin(); it != iod.schemes.ls.dataMap.end(); it++) {
    int matid = it->second->materialid;
    if(matid<=0 || matid>=(int)vf.size()) { //cannot use ls to track material 0
      print_error("*** Error: Cannot initialize a level set for tracking material %d.\n", matid);
      exit_mpi();
    }
    if(ls_tracker.find(matid) != ls_tracker.end()) {
      print_error("*** Error: Cannot initialize multiple level sets for the same material (id=%d).\n", matid);
      exit_mpi();
    }
    ls_tracker.insert(matid);    
    lso[it->first] = new LevelSetOperator(comm, dms, iod, *it->second, spo);
    Phi[it->first] = new SpaceVariable3D(comm, &(dms.ghosted1_1dof));
    NPhi[it->first] = new SpaceVariable3D(comm, &(dms.ghosted1_3dof));
    KappaPhi[it->first] = new SpaceVariable3D(comm, &(dms.ghosted1_1dof));
  }
  // check for user error
  for(int ls=0; ls<OutputData::MAXLS; ls++)
    if(iod.output.levelset[ls] == OutputData::ON && ls>=(int)Phi.size()) {
      print_error("*** Error: Cannot output level set %d, which is undefined.\n"); exit_mpi();}


#ifdef LEVELSET_TEST
  print("\n");
  if(lso.empty()) {
    print_error("*** Error: Activated LEVELSET_TEST (%d), but LevelSetOperator is not specified.\n",
                (int)LEVELSET_TEST);
    exit_mpi();
  }
  print("\033[0;32m- Testing the Level Set Solver using a prescribed velocity field (%d). "
        "N-S solver not activated.\033[0m\n", (int)LEVELSET_TEST);
#endif
  

  // ------------------------------------------------------------------------
  //! Initialize V, ID, Phi. 
  SpaceInitializer spinit(comm, dms, iod, global_mesh, spo.GetMeshCoordinates());
  std::multimap<int, std::pair<int,int> >
  id2closure = spinit.SetInitialCondition(V, ID, Phi, NPhi, KappaPhi, spo, lso, ghand,
                                          embed ? embed->GetPointerToEmbeddedBoundaryData() : nullptr);

  // Boundary conditions are applied to V and Phi. But the ghost nodes of ID have not been populated.
  // ------------------------------------------------------------------------

  //! Initialize multiphase operator (for updating "phase change")
  MultiPhaseOperator mpo(comm, dms, iod, vf, global_mesh, spo, lso);
  if((int)lso.size()>1) { //at each node, at most one "phi" can be negative
    int overlap = mpo.CheckLevelSetOverlapping(Phi);
    if(overlap>0) {
      print_error("*** Error: Found overlapping material subdomains. Number of overlapped cells "
                  "(including duplications): %d.\n", overlap);
      exit_mpi();
    }
  }
  mpo.UpdateMaterialIDAtGhostNodes(ID); //ghost nodes (outside domain) get the ID of their image nodes

  //! Initialize output
  Output out(comm, dms, iod, global_mesh, spo.GetPointerToOuterGhostNodes(), vf, laser, spo.GetMeshCoordinates(),
             spo.GetMeshDeltaXYZ(), spo.GetMeshCellVolumes()); 
  out.InitializeOutput(spo.GetMeshCoordinates());


  //! Initialize time integrator
  TimeIntegratorBase *integrator = TimeIntegratorFE(comm, iod, dms, spo, lso, mpo, nullptr, nullptr, nullptr, nullptr);

  /*************************************
   * Main Loop 
   ************************************/
  print("\n");
  print("----------------------------\n");
  print("--       Main Loop        --\n");
  print("----------------------------\n");
  double t = 0.0; //!< simulation (i.e. physical) time
  double dt = 0.0;
  double cfl = 0.0;
  int time_step = 0;

  //! write initial condition to file
  out.OutputSolutions(t, dt, time_step, V, ID, Phi, NPhi, KappaPhi, nullptr, nullptr, nullptr, true/*force_write*/);

  // find maxTime, and dts (meaningful only with concurrent programs)
  double tmax = iod.ts.maxTime;
  double dts = 0.0;

  // set max time-step number to user input, or INF if it is a follower in Chimera
  int maxIts = iod.ts.maxIts;

  // Time-Stepping
  while(t<tmax-1.0e-6*dts && time_step<maxIts && !integrator->Converged()) {// the last one is for steady-state

    time_step++;
    spo.ComputeTimeStepSize(V, ID, dts, cfl, LocalDt); 

    if(t+dt > tmax) { //update dt at the LAST time step so it terminates at tmax
      cfl *= (tmax - t)/dt;
      dt = tmax - t;
      dtleft = dts;
    }

    print("Step %d: t = %e, dt = %e, cfl = %.4e. Computation time: %.4e s.\n", 
          time_step, t, dts, cfl, walltime()-start_time);
    
    t += dts;
    //TODO: This subscycle needs update
    integrator->AdvanceOneTimeStep(V, ID, Phi, NPhi, KappaPhi, nullptr, nullptr, nullptr, nullptr, t, dts, time_step,
                                   0.0, dts); 

    out.OutputSolutions(t, dts, time_step, V, ID, Phi, NPhi, KappaPhi, nullptr, nullptr, nullptr, false/*force_write*/);

  }

  out.OutputSolutions(t, dts, time_step, V, ID, Phi, NPhi, KappaPhi, nullptr, nullptr, nullptr, true/*force_write*/);

  print("\n");
  print("\033[0;32m==========================================\033[0m\n");
  print("\033[0;32m   NORMAL TERMINATION (t = %e)  \033[0m\n", t); 
  print("\033[0;32m==========================================\033[0m\n");
  print("Total Computation Time: %f sec.\n", walltime()-start_time);
  print("\n");



  //! finalize 
  //! In general, "Destroy" should be called for classes that store Petsc DMDA data (which need to be "destroyed").
  
  //concurrent.Destroy();

  V.Destroy();
  ID.Destroy();

  spinit.Destroy();

  //! Detroy the levelsets
  for(int ls = 0; ls<(int)lso.size(); ls++) {
    Phi[ls]->Destroy(); delete Phi[ls];
    lso[ls]->Destroy(); delete lso[ls];
    NPhi[ls]->Destroy(); delete NPhi[ls];
    KappaPhi[ls]->Destroy(); delete KappaPhi[ls];
  }

  out.FinalizeOutput();
  integrator->Destroy();
  mpo.Destroy();
  spo.Destroy();

  if(grad) {
    grad->Destroy();
    delete grad;
  }
  if(interp) {
    interp->Destroy();
    delete interp;
  }

  dms.DestroyAllDataManagers();

  delete integrator;
  delete ff;
  delete riemann;

  for(int i=0; i<(int)vf.size(); i++)
    delete vf[i];

  PetscFinalize();
  MPI_Finalize();

  return 0;
}

//--------------------------------------------------------------

