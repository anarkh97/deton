/************************************************************************
 * Copyright © 2020 The Multiphysics Modeling and Computation (M2C) Lab
 * <kevin.wgy@gmail.com> <kevinw3@vt.edu>
 ************************************************************************/

#ifndef _VAR_FCN_JWL_BURN_H
#define _VAR_FCN_JWL_BURN_H

#include <VarFcnBase.h>
#include <fstream>
#include <boost/math/tools/roots.hpp>
using namespace boost::math::tools;

/********************************************************************************
 * This class is the VarFcn class for the JWL equation of state (EOS)
 * Only elementary functions are declared and/or defined here.
 * All arguments must be pertinent to only a single grid node or a single
 * state.
 *
 * EOS: Pressure = omega*rho*(e+edet) + f(rho)
 *
 * where
 *   e  : internal energy per unit mass.
 *
 *   f(rho) = A1*(1-omega*rho/(R1*rho0))*exp(-R1*rho0/rho) 
 *          + A2*(1-omega*rho/(R2*rho0))*exp(-R2*rho0/rho)
 *
 *   omega, A1, A2, R1, R2, rho0, edet : constant coefficients (rho0 is the 
 *                                       density of the explosive before denotation, 
 *                                       hence rho0>=rho)
 *
 * References: Arthur Rallu's thesis; Ralph Menikoff's technical report (2016)
 * 
 *   TODO: the temperature law will be implemented later. Ref. Ralph Menikoff, 2016
 *
 ********************************************************************************/
class VarFcnJWLBurn : public VarFcnBase {

private:
  double omega, A1, A2, R1, R2, rho0, edet;

  double R1rho0, R2rho0; //!< R1*rho0, R2*rho0
  double omega_over_R1rho0, omega_over_R2rho0; //!< omega/(R1*rho0), omega/(R2*rho0)

public:
  VarFcnJWLBurn(MaterialModelData &data);
  ~VarFcnJWLBurn() {}

  //! ----- EOS-Specific Functions -----
  inline double GetPressure(double rho, double e) {return omega*rho*(e+edet) + Fun(rho);}
  inline double GetInternalEnergyPerUnitMass(double rho, double p) {return (p-Fun(rho))/(omega*rho) - edet;}
  double GetDensity(double p, double e); 
  double GetDpdrho(double rho, double e); 
  inline double GetBigGamma([[maybe_unused]] double rho, [[maybe_unused]] double e) {return omega;}
  inline double GetTemperature([[maybe_unused]] double rho, [[maybe_unused]] double e) {return 0.0;} //TODO

protected:
  inline double Fun(double rho) {
    return  A1*(1.0-omega_over_R1rho0*rho)*exp(-R1rho0/rho) 
          + A2*(1.0-omega_over_R2rho0*rho)*exp(-R2rho0/rho);}

  //! nested class / functor for root-finding (solving p = JWL(rho,e) for rho)
  struct DensityEquation {

    DensityEquation(double p_, double e_, double edet_, double omegae_, double omegaedet_, double A1_, double A2_,
                    double R1rho0_, double R2rho0_, double omega_over_R1rho0_,
                    double omega_over_R2rho0_)
        : p(p_), e(e_), edet(edet_), omegae(omegae_), omegaedet(omegaedet_), A1(A1_), A2(A2_), 
          R1rho0(R1rho0_), R2rho0(R2rho0_), omega_over_R1rho0(omega_over_R1rho0_), 
          omega_over_R2rho0(omega_over_R2rho0_) {} 

    inline double operator() (double rho) {
      return p - omegae*rho - omegaedet*rho - A1*(1.0-omega_over_R1rho0*rho)*exp(-R1rho0/rho)
                                            - A2*(1.0-omega_over_R2rho0*rho)*exp(-R2rho0/rho);
    }

    private:
    double p, e, edet, omegae, omegaedet, A1, A2, R1rho0, R2rho0, omega_over_R1rho0, omega_over_R2rho0;
  };

};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

VarFcnJWLBurn::VarFcnJWLBurn(MaterialModelData &data) : VarFcnBase(data) {

  if(data.eos != MaterialModelData::JWLBURN){
    fprintf(stdout, "*** Error: MaterialModelData is not of type JWLBurn\n");
    exit(-1);
  }

  type = JWLBURN;

  omega = data.jwlBurnModel.omega;
  A1    = data.jwlBurnModel.A1;
  A2    = data.jwlBurnModel.A2;
  R1    = data.jwlBurnModel.R1;
  R2    = data.jwlBurnModel.R2;
  rho0  = data.jwlBurnModel.rho0;
  edet  = data.jwlBurnModel.edet;

  R1rho0 = R1*rho0;
  R2rho0 = R2*rho0;
  omega_over_R1rho0 = omega/R1rho0;
  omega_over_R2rho0 = omega/R2rho0;

}

//------------------------------------------------------------------------------

double VarFcnJWLBurn::GetDensity(double p, double e) 
{

  DensityEquation equation(p, e, edet, omega*e, omega*edet, A1, A2, R1rho0, 
                           R2rho0, omega_over_R1rho0, omega_over_R2rho0);

  //*******************************************************************
  // Calling boost function for root-finding
  // Warning: "maxit" is BOTH AN INPUT AND AN OUTPUT
  double rho_low = 1e-16; //lowerbound
  double f_low   = equation(rho_low); 
  double rho_hi  = rho0;  //upperbound
  double f_hi    = equation(rho_hi);
  boost::uintmax_t maxit = 500;
  double tol = 1e-8;
  std::pair<double,double> sol = toms748_solve(equation, rho_low, rho_hi, f_low, f_hi,
                                              [=](double r0, double r1){return r1-r0<tol;},
                                              maxit);
  //*******************************************************************

  return 0.5*(sol.first + sol.second);
}

//------------------------------------------------------------------------------

double VarFcnJWLBurn::GetDpdrho(double rho, double e) 
{
  return  omega*(e - edet) 
        + A1*(-omega_over_R1rho0 + R1rho0/(rho*rho) - omega_over_R1rho0*R1rho0/rho)*exp(-R1rho0/rho)
        + A2*(-omega_over_R2rho0 + R2rho0/(rho*rho) - omega_over_R2rho0*R2rho0/rho)*exp(-R2rho0/rho);
}

//------------------------------------------------------------------------------

#endif
