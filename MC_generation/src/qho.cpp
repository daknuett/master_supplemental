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
        std::cerr << "INFO: extra parameters: filename, seed" << std::endl;
        return -1;
    }
    if(argc > 8)
    {
        std::cerr << "WARN: extra arguments (max 5+2)." << std::endl;
    }

    double omega = atof(argv[1]);
    std::size_t n_tau = atol(argv[2]);
    double beta = atof(argv[3]);
    double Delta = atof(argv[4]);
    std::size_t n_markov = atol(argv[5]);

    std::string fname;
    if(argc >= 7)
    {
        fname = argv[6];
    }
    else
    {
        std::stringstream fnstream;
        fnstream << "QHO_data_" << omega << "_" << n_tau << "_" << beta << "_" << Delta << "_" << n_markov << ".bindata";
        fname = fnstream.str();
    }
    if(argc >= 8)
    {
        rne.seed(atol(argv[7]));
    }


    MC::quantum::HOMarkovChain1D chain(n_tau, Delta, beta, omega);
    MC::AutoMC<MC::quantum::HOMarkovChain1D, MC::quantum::observables::observable_t>
        autochain(chain, MC::quantum::observables::position_squared_expect_1D);


    autochain.equilibrate(n_markov / 10, rne);

    std::fstream ofile;
    std::size_t wrote = 0;
    ofile.open(fname, std::ios::out | std::ios::binary);
    for(std::size_t i = 0; i < n_markov; i++)
    {
        autochain.generate_next(rne);
        std::vector<double> * cfg = chain.get_config();
        for(auto v: *cfg)
        {
            ofile.write(reinterpret_cast<char *>(&v), sizeof(v));
            wrote += sizeof(v);
        }
    }

    std::cout << "INFO: wrote " << wrote << " bytes to " << fname << std::endl;

    return 0;
}
