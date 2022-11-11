#include <cstddef>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <MC/quantum.hpp>
#include <MC/automc.hpp>
#include <vector>

int
main(int argc, char ** argv)
{
    std::mt19937_64 rne;
#ifdef USE_DEAD_BEEF
    rne.seed(0xdeadbeef);
#else
    rne.seed(time(0));
#endif

    if(argc < 6)
    {
        std::cerr << "FATAL: missing one of the following parameters: omega^2, n_tau, beta, Delta, n_markov" << std::endl;
        return -1;
    }
    if(argc > 6)
    {
        std::cerr << "WARN: extra arguments (max 5)." << std::endl;
    }

    double omega = atof(argv[1]);
    std::size_t n_tau = atol(argv[2]);
    double beta = atof(argv[3]);
    double Delta = atof(argv[4]);
    std::size_t n_markov = atol(argv[5]);


    std::size_t const n_statistic = 400;

    std::vector<double> tauints;

    for(std::size_t j = 0; j < n_statistic; j++)
    {
        MC::quantum::HOMarkovChain1D chain(n_tau, Delta, beta, omega);
        MC::AutoMC<MC::quantum::HOMarkovChain1D, MC::quantum::observables::observable_t>
            autochain(chain, MC::quantum::observables::position_squared_expect_1D, true);


        autochain.equilibrate(n_markov / 10, rne);

        std::vector<double> x2;
        for(std::size_t i = 0; i < n_markov; i++)
        {
            chain.generate_next(rne);
            x2.push_back(chain.get_observable(MC::quantum::observables::position_squared_expect_1D));
        }

        statistic::Statistic<double> x2stat(x2);
        tauints.push_back(x2stat.integrated_autocorrelation_time());
    }
    statistic::Statistic<double> tauintstat(tauints);
    std::cout << tauintstat.get_avg() << "," << tauintstat.get_std() << std::endl;


    return 0;
}

