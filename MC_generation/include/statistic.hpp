#pragma once
#include <vector>
#include <cmath>

namespace statistic
{
    template <typename T>
    class Statistic
    {
        private:
            std::vector<T> data;

        public:
            template <typename C>
            double
            integrate_over(C fct)
            {
                int n = 0;
                double result = T{0};
                for(T datapoint: data)
                {
                    result += fct(datapoint);
                    n++;
                }
                return result / (double)n;
            }

            template <typename C>
            double
            integrate_over(C fct, std::vector<int> & select)
            {
                int n = 0;
                double result = {0};
                for(int idx: select)
                {
                    result += fct(data.at(idx));
                    n++; }
                return result / (double)n;
            }
            template <typename C, typename N>
            double
            integrate_over(C fct, N normalizer)
            {
                int n = 0;
                double result = T{0};
                for(T datapoint: data)
                {
                    result += fct(datapoint);
                    n++;
                }
                return result / (double)normalizer(n);
            }

            template <typename C, typename N>
            double
            integrate_over(C fct
                        , std::vector<int> & select
                        , N normalizer)
            {
                int n = 0;
                double result = {0};
                for(int idx: select)
                {
                    result += fct(data.at(idx));
                    n++; }
                return result / (double)normalizer(n);
            }
            Statistic(std::vector<T> & vct):
                data(vct)
            {

            }

            size_t
            get_data_length(void)
            {
                return data.size();
            }

            double
            get_avg(std::vector<int> & select)
            {
                return integrate_over([](T val){
                        return val;
                        }, select);
            }

            double
            get_avg()
            {
                return integrate_over([](T val)
                        {
                            return val;
                        });
            }

            double
            get_var(std::vector<int> & select)
            {
                double avg = get_avg(select);
                double x_squared_avg = integrate_over([](T val)
                        {
                            return val*val;
                        }, select);
                return x_squared_avg - avg*avg;
            }

            double
            get_var()
            {
                double avg = get_avg();
                double x_squared_avg = integrate_over([](T val)
                        {
                            return val*val;
                        });
                return x_squared_avg - avg*avg;
            }

            double 
            get_std()
            {
                return std::sqrt(std::abs(get_var()));
            }

            double
            get_std(std::vector<int> & select)
            {
                return std::sqrt(std::abs(get_var(select)));
            }

            template <typename C>
            double
            expectation_value(C f)
            {
                return integrate_over<C>(f);
            }

            template <typename C>
            double
            variance(C f)
            {
                double expv = expectation_value<C>(f);
                return integrate_over(
                            [f](T x)
                            {
                                return f(x) * f(x);
                            }
                        ) - expv * expv;
            }

            template <typename C>
            double
            expectation_value(C f, std::vector<int> & select)
            {
                return integrate_over<C>(f, select);
            }

            template <typename C>
            double
            variance(C f, std::vector<int> & select)
            {
                double expv = expectation_value<C>(f, select);
                return integrate_over(
                            [f](T x)
                            {
                                return f(x) * f(x);
                            }
                            , select
                        ) - expv * expv;
            }

            template <typename C>
            double
            get_sample_mean(C f)
            {
                return integrate_over(f);
            }

            template <typename C>
            double
            get_sample_std(C f)
            {
                double expv = get_sample_mean(f);
                return std::sqrt(integrate_over([f, expv](T x)
                        {
                            double v = f(x) - expv;
                            return v * v;
                        }
                        , [](int n)
                        {
                            return n - 1;
                        }
                        ));
            }
            template <typename C>
            double
            get_sample_mean(C f, std::vector<int> & select)
            {
                return integrate_over(f, select);
            }

            template <typename C>
            double
            get_sample_std(C f, std::vector<int> & select)
            {
                double expv = get_sample_mean(f, select);
                return std::sqrt(integrate_over([f, expv](T x)
                        {
                            double v = f(x) - expv;
                            return v * v;
                        }
                        , select
                        , [](int n)
                        {
                            return n - 1;
                        }
                        ));
            }

            double
            autocovariance_function(int t)
            {
                std::vector<int> select_p, select_m;
                auto N = get_data_length();
                for(std::size_t i = 0; i < N - t; i++)
                {
                    select_m.push_back(i);
                    select_p.push_back(i + t);
                }

                auto ybar_m = get_avg(select_m);
                auto ybar_p = get_avg(select_p);

                double res = 0;
                for(std::size_t i = 0; i < N - t; i++)
                {
                    res += (data[i] - ybar_m) * (data[i + t] - ybar_p);
                }

                return res / (N - t);
            }

            double
            autocovariance_function(int t, std::vector<int> & select)
            {
                std::vector<int> select_p, select_m;
                auto N = select.size();
                for(auto i = 0; i < N - t; i++)
                {
                    select_m.push_back(select[i]);
                    select_p.push_back(select[i + t]);
                }

                auto ybar_m = get_avg(select_m);
                auto ybar_p = get_avg(select_p);

                double res = 0;
                for(auto i = 0; i < N - t; i++)
                {
                    res += (data[select[i]] - ybar_m) * (data[select[i + t]] - ybar_p);
                }

                return res / (N - t);
            }

            template <typename C>
            double
            autocovariance_function(C f, int t)
            {
                std::vector<int> select_p, select_m;
                auto N = get_data_length();
                for(auto i = 0; i < N - t; i++)
                {
                    select_m.push_back(i);
                    select_p.push_back(i + t);
                }

                auto ybar_m = get_sample_mean(f, select_m);
                auto ybar_p = get_sample_mean(f, select_p);

                double res = 0;
                for(auto i = 0; i < N - t; i++)
                {
                    res += (data[i] - ybar_m) * (data[i + t] - ybar_p);
                }

                return res / (N - t);
            }

            template <typename C>
            double
            autocovariance_function(C f, int t, std::vector<int> & select)
            {
                std::vector<int> select_p, select_m;
                auto N = select.size();
                for(auto i = 0; i < N - t; i++)
                {
                    select_m.push_back(select[i]);
                    select_p.push_back(select[i + t]);
                }

                auto ybar_m = get_sample_mean(f, select_m);
                auto ybar_p = get_sample_mean(f, select_p);

                double res = 0;
                for(auto i = 0; i < N - t; i++)
                {
                    res += (data[select[i]] - ybar_m) * (data[select[i + t]] - ybar_p);
                }

                return res / (N - t);
            }

            double
            integrated_autocorrelation_time(void)
            {
                double zero_autocov = autocovariance_function(0);
                double result = 0;
                for(std::size_t i = 1; i < get_data_length(); i++)
                {
                    double autocorr = autocovariance_function(i);
                    if(autocorr < 0)
                    {
                        break;
                    }
                    result += autocorr;
                }
                return result / zero_autocov + .5;
            }

            double
            integrated_autocorrelation_time(std::vector<int> & select)
            {
                double zero_autocov = autocovariance_function(0, select);
                double result = 0;
                for(auto i = 1; i < select.size(); i++)
                {
                    double autocorr = autocovariance_function(i, select);
                    if(autocorr < 0)
                    {
                        break;
                    }
                    result += autocorr;
                }
                return result / zero_autocov + .5;
            }

            template <typename C>
            double
            integrated_autocorrelation_time(C f)
            {
                double zero_autocov = autocovariance_function(f, 0);
                double result = 0;
                for(auto i = 1; i < get_data_length(); i++)
                {
                    double autocorr = autocovariance_function(f, i);
                    if(autocorr < 0)
                    {
                        break;
                    }
                    result += autocorr;
                }
                return result / zero_autocov + .5;
            }

            template <typename C>
            double
            integrated_autocorrelation_time(C f, std::vector<int> & select)
            {
                double zero_autocov = autocovariance_function(f, 0, select);
                double result = 0;
                for(auto i = 1; i < select.size(); i++)
                {
                    double autocorr = autocovariance_function(f, i, select);
                    if(autocorr < 0)
                    {
                        break;
                    }
                    result += autocorr;
                }
                return result / zero_autocov + .5;
            }

            double
            standard_error_autocorrelated(void)
            {
                return get_std() * std::sqrt(2*integrated_autocorrelation_time() / get_data_length());
            }

            double
            standard_error_autocorrelated(std::vector<int> & select)
            {
                return get_std(select) * std::sqrt(2*integrated_autocorrelation_time(select) / select.size());
            }

            template <typename C>
            double
            standard_error_autocorrelated(C f)
            {
                return get_sample_std(f) * std::sqrt(2*integrated_autocorrelation_time(f) / get_data_length());
            }

            template <typename C>
            double
            standard_error_autocorrelated(C f, std::vector<int> & select)
            {
                return get_sample_std(f, select) * std::sqrt(2*integrated_autocorrelation_time(f, select) / select.size());
            }
    };
}
