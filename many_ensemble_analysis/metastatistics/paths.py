from pathlib import Path

# lower precision
# Use this for autocorrelation measurement
out_path_unmod_lp = Path("XXXXX/HO_configs_meta_unmod_lp/")
meta_unmod_lp = dict(n_tau = 40
                , n_markov = 750000
                , omega = 0.5
                , beta = 10
                , Delta = 1
                )

# lower precision
out_path_mod_lp = Path("XXXXX/HO_configs_meta_mod_lp/")
meta_mod_lp = dict(n_tau = 40
                , n_markov = 750000
                , omega = 0.5
                , beta = 10
                , Delta = 1
                , tinsert = 0
                , Eb = 3.4867
                )


# higher precision.
#out_path_unmod = Path("XXXXX/HO_configs_meta_unmod_hp/")
out_path_unmod = Path("XXXXX/HO_configs_meta_unmod_hp/")
meta_unmod = dict(n_tau = 40
                , n_markov = 8_000_000
                , omega = 0.5
                , beta = 10
                , Delta = 1
                )

# higher precision
out_path_mod = Path("XXXXX/HO_configs_meta_mod_hp_ses/")
meta_mod = dict(n_tau = 40
                , n_markov = 8_000_000
                , omega = 0.5
                , beta = 10
                , Delta = 1
                , tinsert = 0
                , Eb = 3.4867
                )

# higher precision
#out_path_mod_gs = Path("XXXXX/HO_configs_meta_mod_hp_sgs/")
out_path_mod_gs = Path("XXXXX/HO_configs_meta_mod_hp_sgs/")
meta_mod_gs = dict(n_tau = 40
                , n_markov = 8_000_000
                , omega = 0.5
                , beta = 10
                , Delta = 1
                , tinsert = 0
                , Eb = 1.49269
                )
