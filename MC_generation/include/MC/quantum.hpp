#pragma once
#include <vector>
#include <random>

namespace MC { 
    namespace quantum
    {
        class HOMarkovChain1D
        {
            protected:
            std::size_t m_ntau;
            std::vector<double> * m_Xi;
            double m_delta;
            double m_beta;
            double m_omega2;
            virtual inline double action_from_site(std::size_t site);
            virtual inline double action_from_site_with_replacement(std::size_t site, double replacement);
            public:
            HOMarkovChain1D(std::size_t ntau, double delta, double beta, double omega2);
            virtual ~HOMarkovChain1D(void)
            {
                delete m_Xi;
            }
            virtual std::size_t ntau(void)
            {
                return m_ntau;
            }
            template <typename T>
            double get_observable(T observable)
            {
                return observable(m_Xi);
            }
            virtual int generate_next(std::mt19937_64 & rne);
            
            std::vector<double> * get_config()
            {
                return m_Xi;
            }
            double get_correlator(std::size_t t);

        };

        class DeltaTHHOMarkovChain1D: public HOMarkovChain1D
        {
            protected:
            std::size_t m_tinsert;
            double m_Ebias;
            inline double site_probability(std::size_t i);
            inline double site_probability_with_replacement(double candidate, std::size_t i);
            inline double reweight_factor(void);
            inline double reweight_factor_with_replacement(double candidate, std::size_t j);
            public:
            DeltaTHHOMarkovChain1D(std::size_t ntau, double delta, double beta, double omega2, std::size_t tinsert, double Ebias):
                HOMarkovChain1D(ntau, delta, beta, omega2), m_tinsert(tinsert), m_Ebias(Ebias)
            {}
            int generate_next(std::mt19937_64 & rne);
        };

        namespace observables
        {
            typedef double (* observable_t)(std::vector<double> * config);
            double position_expect_1D(std::vector<double> * config);
            double position_squared_expect_1D(std::vector<double> * config);
            double action_HO_1D(std::vector<double> * config, double omega, double a);
        }
    } 
}
