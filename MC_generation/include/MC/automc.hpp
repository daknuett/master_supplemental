#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <statistic.hpp>

namespace MC
{
    template <typename MC_t, typename observable_t>
    class AutoMC
    {
        protected:
        MC_t & m_chain;
        observable_t m_observable;
        bool m_is_equilibrated;
        int m_autocorr;
        bool m_bquiet;
        unsigned long long int m_ngenerated;

        void try_equilibration(std::size_t n_try, std::mt19937_64 & rne)
        {
            for(std::size_t i = 0; i < n_try; i++)
            {
                m_chain.generate_next(rne);
                m_ngenerated++;
            }
        }

        public:
        unsigned long long int ngenerated(void)
        {
            return m_ngenerated;
        }
        int get_autocorr(void)
        {
            return m_autocorr;
        }
        AutoMC(MC_t & chain, observable_t observable):
            m_chain(chain), m_observable(observable)
        {
            m_is_equilibrated = false;
            m_bquiet = false;
            m_ngenerated = 0;
        }
        AutoMC(MC_t & chain, observable_t observable, bool bequiet):
            m_chain(chain), m_observable(observable), m_bquiet(bequiet)
        {
            m_is_equilibrated = false;
        }
        double generate_next(std::mt19937_64 & rne)
        {
            double accepted = 0;
            for(int i = 0; i < m_autocorr; i++)
            {
                accepted += m_chain.generate_next(rne);
                m_ngenerated++;
            }
            return accepted / m_chain.ntau() / m_autocorr;
        }
        template <typename C>
        double get_observable(C observable)
        {
            return m_chain.get_observable(observable);
        }
        std::vector<double> * get_config()
        {
            return m_chain.get_config();
        }
        void equilibrate(std::size_t equilibration_expect, std::mt19937_64 & rne)
        {
            try_equilibration(equilibration_expect, rne);
            std::vector<double> observable;
            double accepted = 0;
            for(std::size_t i = 0; i < equilibration_expect; i++)
            {
                accepted += m_chain.generate_next(rne);
                m_ngenerated++;
                observable.push_back(m_chain.get_observable(m_observable));
            }
            double accceptance_rate = accepted / (m_chain.ntau() * equilibration_expect);
            if(accceptance_rate < 0.2 || accceptance_rate > 0.8)
            {
                std::cerr << "AutoMC: accceptance rate is " << accceptance_rate << ". Estimates for autocorrelation time are probably wrong." << std::endl;
                std::cerr << "AutoMC: Consider changing your step parameter." << std::endl;
            }

            statistic::Statistic<double> ostat(observable);
            double tau_int = ostat.integrated_autocorrelation_time();
            if(not m_bquiet)
            {
                std::cerr << "AutoMC: tau_int (1st estimate): " << tau_int << std::endl;
            }

            if(tau_int * 30 > equilibration_expect)
            {
                if(not m_bquiet)
                {
                    std::cerr << "AutoMC: found 30*tau_int > equilibration_expect. Re-equilibrating." << std::endl;
                    std::cerr << "AutoMC: re-equilibrating with " << (int) (100*tau_int) << " MC steps." << std::endl;
                }
                try_equilibration((std::size_t) tau_int * 100, rne);
                equilibration_expect = 20 * tau_int;
            }

            observable.resize(0);
            accepted = 0;
            for(std::size_t i = 0; i < equilibration_expect; i++)
            {
                accepted += m_chain.generate_next(rne);
                m_ngenerated++;
                observable.push_back(m_chain.get_observable(m_observable));
            }
            accceptance_rate = accepted / (m_chain.ntau() * equilibration_expect);
            if(accceptance_rate < 0.2 || accceptance_rate > 0.8)
            {
                std::cerr << "AutoMC: (After equilibration): accceptance rate is " << accceptance_rate << ". Estimates for autocorrelation time are probably wrong." << std::endl;
                std::cerr << "AutoMC: (After equilibration): Consider changing your step parameter." << std::endl;
            }

            statistic::Statistic<double> ostat_new(observable);
            tau_int = ostat_new.integrated_autocorrelation_time();
            if(not m_bquiet)
            {
                std::cerr << "AutoMC: tau_int (2nd estimate): " << tau_int << std::endl;
            }
            m_autocorr = 2*tau_int;
            if(m_autocorr < 1)
            {
                m_autocorr = 1;
            }
            if(not m_bquiet)
            {
                std::cerr << "AutoMC: working with 2*tau_int: " << m_autocorr << std::endl;
            }
            m_is_equilibrated = true;
        }
    };
}
