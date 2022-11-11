#include <MC/quantum.hpp>
#include <iostream>

namespace MC
{
    namespace quantum
    {
        HOMarkovChain1D::HOMarkovChain1D(std::size_t ntau, double delta, double beta, double omega2):
            m_ntau(ntau), m_delta(delta), m_beta(beta), m_omega2(omega2)
        {
            m_Xi = new std::vector<double>(m_ntau, 0);
        }

        inline double HOMarkovChain1D::action_from_site(std::size_t site)
        {
            std::size_t N = m_ntau;
            std::size_t ip = (site + 1) % m_ntau;
            std::size_t im = (site + m_ntau - 1) % m_ntau;

            return ((*m_Xi)[site] * ((*m_Xi)[site] - (*m_Xi)[ip] - (*m_Xi)[im]) * N / m_beta
                    + m_omega2 * (*m_Xi)[site] * (*m_Xi)[site] * m_beta / N);
        }
        inline double HOMarkovChain1D::action_from_site_with_replacement(std::size_t site, double replacement)
        {
            std::size_t N = m_ntau;
            std::size_t ip = (site + 1) % m_ntau;
            std::size_t im = (site + m_ntau - 1) % m_ntau;

            return (replacement * (replacement - (*m_Xi)[ip] - (*m_Xi)[im]) * N / m_beta 
                    + m_omega2 *  replacement * replacement * m_beta / N);
        }


        int HOMarkovChain1D::generate_next(std::mt19937_64 & rne)
        {
            int accepted = 0;
            std::uniform_real_distribution<double> rng(-m_delta, m_delta);
            std::uniform_real_distribution<double> accept_rng(0, 1);

            std::size_t const cachline_size = 64;
            std::size_t const chunkify = 4;
            std::size_t const objs_per_cacheline = cachline_size / sizeof(double);
            std::size_t cachelines = m_ntau / objs_per_cacheline;
            if(m_ntau % objs_per_cacheline)
            {
                cachelines++;
            }
            
            // By default (see meson.build) we disable the parallelization of the
            // in-MC loops because the overhead exceeds the benefit for small lattices.
            // for lattice sizes > 1k one should enable the parallelization.
            
            // This loop ensures that no there are no cache line collisions
#ifndef DISABLE_INMC_PARALLEL
#pragma omp parallel
#endif
            for(std::size_t start_from = 0; start_from < chunkify; start_from++)
            {
                // Loop over the cache lines.
#ifndef DISABLE_INMC_PARALLEL
#pragma omp for nowait
#endif
                for(std::size_t i_chunk = start_from; i_chunk < cachelines; i_chunk += chunkify)
                {
                    std::size_t chunk = i_chunk * objs_per_cacheline;
                    // Loop through the cache line.
                    for(std::size_t i = chunk; i < chunk + objs_per_cacheline && i < m_ntau; i++)
                    {
                        double candidate = (*m_Xi)[i] + rng(rne);

                        //if(delta_S < 0)
                        //{
                        //    accepted++;
                        //    (*m_Xi)[i] = candidate;
                        //    continue;
                        //}
#ifdef USE_FRACT_PROBABILITY
                        double probability = std::exp(-action_from_site_with_replacement(i, candidate)) / std::exp(-action_from_site(i));
#else

                        double delta_S = action_from_site_with_replacement(i, candidate)
                                        - action_from_site(i);
                        double probability = std::exp(- delta_S);
#endif
                        if(accept_rng(rne) <  probability)
                        {
                            accepted++;
                            (*m_Xi)[i] = candidate;
                            continue;
                        }
                    }
                }
            }

            return accepted;
        }        

        double HOMarkovChain1D::get_correlator(std::size_t t)
        {
            double res = 0;
            for(std::size_t s = 0; s < m_ntau; s++)
            {
                res += (*m_Xi)[s] * (*m_Xi)[(t + s) % m_ntau];
            }
            return res / m_ntau;
        }

        
        int DeltaTHHOMarkovChain1D::generate_next(std::mt19937_64 & rne) 
        {
            int accepted = 0;
            std::uniform_real_distribution<double> rng(-m_delta, m_delta);
            std::uniform_real_distribution<double> accept_rng(0, 1);

            std::size_t const cachline_size = 64;
            std::size_t const chunkify = 4;
            std::size_t const objs_per_cacheline = cachline_size / sizeof(double);
            std::size_t cachelines = m_ntau / objs_per_cacheline;
            if(m_ntau % objs_per_cacheline)
            {
                cachelines++;
            }
            
            // By default (see meson.build) we disable the parallelization of the
            // in-MC loops because the overhead exceeds the benefit for small lattices.
            // for lattice sizes > 1k one should enable the parallelization.
            
            // This loop ensures that no there are no cache line collisions
#ifndef DISABLE_INMC_PARALLEL
#pragma omp parallel
#endif
            for(std::size_t start_from = 0; start_from < chunkify; start_from++)
            {
                // Loop over the cache lines.
#ifndef DISABLE_INMC_PARALLEL
#pragma omp for nowait
#endif
                for(std::size_t i_chunk = start_from; i_chunk < cachelines; i_chunk += chunkify)
                {
                    std::size_t chunk = i_chunk * objs_per_cacheline;
                    // Loop through the cache line.
                    for(std::size_t i = chunk; i < chunk + objs_per_cacheline && i < m_ntau; i++)
                    {
                        double candidate = (*m_Xi)[i] + rng(rne);

                        double probability = site_probability_with_replacement(candidate, i) / site_probability(i);
                        if(accept_rng(rne) <  probability)
                        {
                            accepted++;
                            (*m_Xi)[i] = candidate;
                            continue;
                        }
                    }
                }
            }

            return accepted;
        }

        inline double DeltaTHHOMarkovChain1D::site_probability_with_replacement(double candidate, std::size_t i)
        {
            double S_E = action_from_site_with_replacement(i, candidate);
            if(i != m_tinsert && i != m_tinsert + 1)
            {
                return std::exp(-S_E);
            }
            //std::cerr << "site: " << i << " extra factor: " << std::abs(reweight_factor_with_replacement(candidate, i) - m_Ebias*m_beta/m_ntau) << std::endl;
            return std::exp(-S_E) * std::abs(reweight_factor_with_replacement(candidate, i) - m_Ebias*m_beta/m_ntau);
            
        }
        inline double DeltaTHHOMarkovChain1D::site_probability(std::size_t i)
        {
            double S_E = action_from_site(i);
            if(i != m_tinsert && i != m_tinsert + 1)
            {
                return std::exp(-S_E);
            }
            //std::cerr << "site: " << i << " extra factor: " << std::abs(reweight_factor() - m_Ebias*m_beta/m_ntau) << std::endl;
            return std::exp(-S_E) * std::abs(reweight_factor() - m_Ebias*m_beta/m_ntau);
        }
        inline double DeltaTHHOMarkovChain1D::reweight_factor(void)
        {
            std::size_t i = m_tinsert;
            std::size_t ip = (m_tinsert + 1) % m_ntau;
            return 0.5 - (
                    ((*m_Xi)[ip] - (*m_Xi)[i]) * ((*m_Xi)[ip] - (*m_Xi)[i]) / 2 / m_beta * m_ntau
                    - ((*m_Xi)[ip] * (*m_Xi)[ip] + (*m_Xi)[i] * (*m_Xi)[i]) / 2 * m_omega2 * m_beta / m_ntau
                    );
        }
        inline double DeltaTHHOMarkovChain1D::reweight_factor_with_replacement(double candidate, std::size_t j)
        {
            std::size_t i = m_tinsert;
            std::size_t ip = (m_tinsert + 1) % m_ntau;
            if(i == j)
            {
                return 0.5 - (
                        ((*m_Xi)[ip] - candidate) * ((*m_Xi)[ip] - candidate) / 2 / m_beta * m_ntau
                        - ((*m_Xi)[ip] * (*m_Xi)[ip] + candidate * candidate) / 2 * m_omega2 * m_beta / m_ntau
                        );
            }
            if(ip == j)
            {
                return 0.5 - (
                        (candidate - (*m_Xi)[i]) * (candidate - (*m_Xi)[i]) / 2 / m_beta * m_ntau
                        - (candidate * candidate + (*m_Xi)[i] * (*m_Xi)[i]) / 2 * m_omega2 * m_beta / m_ntau
                        );
            }
            return 0.5 - (
                    ((*m_Xi)[ip] - (*m_Xi)[i]) * ((*m_Xi)[ip] - (*m_Xi)[i]) / 2 / m_beta * m_ntau
                    - ((*m_Xi)[ip] * (*m_Xi)[ip] + (*m_Xi)[i] * (*m_Xi)[i]) / 2 * m_omega2 * m_beta / m_ntau
                    );
        }



        namespace observables
        {
            double position_expect_1D(std::vector<double> * config)
            {
                double result = 0;
                for(auto x: *config)
                {
                    result += x;
                }
                return result / config->size();
            }
            double position_squared_expect_1D(std::vector<double> * config)
            {
                double result = 0;
                for(auto x: *config)
                {
                    result += x*x;
                }
                return result / config->size();
            }
            double action_HO_1D(std::vector<double> * config, double omega, double a)
            {
                double result = 0;
                for(std::size_t i = 0; i < config->size() - 1; i++)
                {
                    std::size_t const ip = (i + 1) % config->size();
                    double deriv_factor = ((*config)[ip] - (*config)[i]);
                    result += ( deriv_factor*deriv_factor / (2 * a)
                                + omega * a * ((*config)[i] * (*config)[i] 
                                                + (*config)[ip] * (*config)[ip]) / 2);
                }
                
                return result;
            }
        }
    }
}
