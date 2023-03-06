PARAMS_DICTS_DRIFT = {
    "reg_enn": {"alpha": 1, "nr_of_neighbors": 3},
    "reg_enn_time": {"alpha": 1, "nr_of_neighbors": 4, "time_scaling_factor": 1000},
    "reg_cnn": {"alpha": 0.05, "nr_of_neighbors": 6},
    "drop_two_re": {"nr_of_neighbors": 5},
    "drop_three_re": {
        "alpha": 0.5,
        "nr_of_neighbors": 7,
        "reg_enn_alpha": 1,
        "reg_enn_neighbors": 3,
    },
    "drop_two_rt": {"alpha": 1, "nr_of_neighbors": 8},
    "shapley": {
        "learning_rate": 1.000000e-04,
        "max_iteration": 30,
        "convergence_error": 0.1,
        "subsize_frac": 0.5,
    },
    "selcon": {"val_frac": 0.25, "subsize_frac": 0.5},
    "lof": {
        "nr_of_neighbors": 7,
    },
    "mutual_information": {"alpha": 0.01, "nr_of_neighbors": 5},
    "fixed_window": {"subsize_frac": 0.5},
    "full": {},
    "ground_truth": {},
}
DRIFT_DEPENDENT_PARAMS = {
    "none": {
        "drop_three_rt": {
            "alpha": 0.25,
            "nr_of_neighbors": 5,
            "reg_enn_alpha": 1,
            "reg_enn_neighbors": 3,
        },
        "fish1": {
            "subsize_frac": 0.5,
            "temporal_weight": 0.5,
        },
    },
    "sudden": {
        "drop_three_rt": {
            "alpha": 0.25,
            "nr_of_neighbors": 5,
            "reg_enn_alpha": 1,
            "reg_enn_neighbors": 3,
        },
        "fish1": {
            "subsize_frac": 0.5,
            "temporal_weight": 0.9,
        },
    },
    "gradual": {
        "drop_three_rt": {
            "alpha": 0.25,
            "nr_of_neighbors": 5,
            "reg_enn_alpha": 1,
            "reg_enn_neighbors": 3,
        },
        "fish1": {
            "subsize_frac": 0.5,
            "temporal_weight": 0.8,
        },
    },
    "increment": {
        "drop_three_rt": {
            "alpha": 0.5,
            "nr_of_neighbors": 7,
            "reg_enn_alpha": 1,
            "reg_enn_neighbors": 3,
        },
        "fish1": {
            "subsize_frac": 0.5,
            "temporal_weight": 0.7,
        },
    },
    "reoccurring": {
        "drop_three_rt": {
            "alpha": 1,
            "nr_of_neighbors": 9,
            "reg_enn_alpha": 1,
            "reg_enn_neighbors": 3,
        },
        "fish1": {
            "subsize_frac": 0.5,
            "temporal_weight": 0.6,
        },
    },
}
PARAMS_DICTS_NOISE = {
    "reg_enn": {"alpha": 1, "nr_of_neighbors": 3},
    "reg_enn_time": {
        "alpha": 1,
        "nr_of_neighbors": 3,
        "time_scaling_factor": 300,
        "distance_measure": "linear",
    },
    "reg_cnn": {"alpha": 0.05, "nr_of_neighbors": 3},
    "drop_two_rt": {"alpha": 1, "nr_of_neighbors": 8},
    "drop_two_re": {"nr_of_neighbors": 3},
    "drop_three_re": {
        "alpha": 0.5,
        "nr_of_neighbors": 3,
        "reg_enn_alpha": 1,
        "reg_enn_neighbors": 3,
    },
    "drop_three_rt": {
        "alpha": 0.25,
        "nr_of_neighbors": 9,
        "reg_enn_alpha": 1,
        "reg_enn_neighbors": 3,
    },
    "lof": {
        "nr_of_neighbors": 3,
    },
    "mutual_information": {"alpha": 0.01, "nr_of_neighbors": 3},
    "fixed_window": {"subsize_frac": 0.8},
}
NOISE_DEPENDENT_PARAMS = {
    0.1: {
        "shapley": {
            "learning_rate": 1.000000e-06,
            "max_iteration": 40,
            "convergence_error": 0.05,
            "subsize_frac": 0.8,
        },
        "fish1": {"subsize_frac": 0.9, "temporal_weight": 0},
        "selcon": {"val_frac": 0.25, "subsize_frac": 0.9},
        "random": {"subsize_frac": 0.9},
    },
    0.2: {
        "shapley": {
            "learning_rate": 1.000000e-06,
            "max_iteration": 100,
            "convergence_error": 0.05,
            "subsize_frac": 0.8,
        },
        "fish1": {"subsize_frac": 0.8, "temporal_weight": 0},
        "selcon": {"val_frac": 0.25, "subsize_frac": 0.8},
        "random": {"subsize_frac": 0.8},
    },
    0.3: {
        "shapley": {
            "learning_rate": 1.000000e-06,
            "max_iteration": 150,
            "convergence_error": 0.1,
            "subsize_frac": 0.7,
        },
        "fish1": {"subsize_frac": 0.7, "temporal_weight": 0},
        "selcon": {"val_frac": 0.25, "subsize_frac": 0.8},
    },
    "random": {"subsize_frac": 0.7},
}
PARAMS_DICT_COMPANIES = {
    "reg_enn": {
        "amoeneburg": {"alpha": 5, "nr_of_neighbors": 4},
        "maerker": {"alpha": 5, "nr_of_neighbors": 3},
        "rohrdorfer": {"alpha": 3, "nr_of_neighbors": 8},
        "spenner": {"alpha": 5, "nr_of_neighbors": 4},
        "woessingen": {"alpha": 5, "nr_of_neighbors": 3},
    },  # from Arnaiz paper Reg by discretization, alpha should be bigger 1
    "reg_enn_time": {
        "amoeneburg": {
            "alpha": 5,
            "nr_of_neighbors": 7,
            "time_scaling_factor": 750,
            "distance_measure": "exp",
        },
        "maerker": {
            "alpha": 5,
            "nr_of_neighbors": 3,
            "time_scaling_factor": 500,
            "distance_measure": "exp",
        },
        "rohrdorfer": {
            "alpha": 3,
            "nr_of_neighbors": 8,
            "time_scaling_factor": 500,
            "distance_measure": "exp",
        },
        "spenner": {
            "alpha": 5,
            "nr_of_neighbors": 4,
            "time_scaling_factor": 500,
            "distance_measure": "exp",
        },
        "woessingen": {
            "alpha": 5,
            "nr_of_neighbors": 3,
            "time_scaling_factor": 500,
            "distance_measure": "exp",
        },
    },
    "reg_cnn": {
        "amoeneburg": {"alpha": 0.05, "nr_of_neighbors": 4},
        "maerker": {"alpha": 0.05, "nr_of_neighbors": 5},
        "rohrdorfer": {"alpha": 0.05, "nr_of_neighbors": 5},
        "spenner": {"alpha": 0.05, "nr_of_neighbors": 5},
        "woessingen": {"alpha": 0.05, "nr_of_neighbors": 8},
    },  # from Arnaiz paper DROP
    "drop_two_re": {
        "amoeneburg": {"nr_of_neighbors": 6},
        "maerker": {"nr_of_neighbors": 6},
        "rohrdorfer": {"nr_of_neighbors": 8},
        "spenner": {"nr_of_neighbors": 5},
        "woessingen": {"nr_of_neighbors": 6},
    },
    "drop_two_rt": {
        "amoeneburg": {"alpha": 0.5, "nr_of_neighbors": 6},
        "maerker": {"alpha": 1, "nr_of_neighbors": 6},
        "rohrdorfer": {"alpha": 0.5, "nr_of_neighbors": 4},
        "spenner": {"alpha": 0.5, "nr_of_neighbors": 8},
        "woessingen": {"alpha": 0.5, "nr_of_neighbors": 5},
    },
    "drop_three_re": {
        "amoeneburg": {
            "alpha": 0.5,
            "nr_of_neighbors": 6,
            "reg_enn_alpha": 5,
            "reg_enn_neighbors": 4,
        },
        "maerker": {
            "alpha": 0.05,
            "nr_of_neighbors": 9,
            "reg_enn_alpha": 5,
            "reg_enn_neighbors": 3,
        },
        "rohrdorfer": {
            "alpha": 0.5,
            "nr_of_neighbors": 8,
            "reg_enn_alpha": 3,
            "reg_enn_neighbors": 8,
        },
        "spenner": {
            "alpha": 1,
            "nr_of_neighbors": 11,
            "reg_enn_alpha": 5,
            "reg_enn_neighbors": 4,
        },
        "woessingen": {
            "alpha": 1,
            "nr_of_neighbors": 6,
            "reg_enn_alpha": 5,
            "reg_enn_neighbors": 3,
        },
    },
    "drop_three_rt": {
        "amoeneburg": {
            "alpha": 1,
            "nr_of_neighbors": 9,
            "reg_enn_alpha": 5,
            "reg_enn_neighbors": 4,
        },
        "maerker": {
            "alpha": 0.5,
            "nr_of_neighbors": 7,
            "reg_enn_alpha": 5,
            "reg_enn_neighbors": 3,
        },
        "rohrdorfer": {
            "alpha": 0.05,
            "nr_of_neighbors": 8,
            "reg_enn_alpha": 3,
            "reg_enn_neighbors": 8,
        },
        "spenner": {
            "alpha": 1,
            "nr_of_neighbors": 9,
            "reg_enn_alpha": 5,
            "reg_enn_neighbors": 4,
        },
        "woessingen": {
            "alpha": 0.5,
            "nr_of_neighbors": 7,
            "reg_enn_alpha": 5,
            "reg_enn_neighbors": 3,
        },
    },
    "selcon": {
        "amoeneburg": {"val_frac": 0.05, "subsize_frac": 0.8},
        "maerker": {"val_frac": 0.1, "subsize_frac": 0.8},
        "rohrdorfer": {"val_frac": 0.05, "subsize_frac": 0.9},
        "spenner": {"val_frac": 0.05, "subsize_frac": 0.9},
        "woessingen": {"val_frac": 0.05, "subsize_frac": 0.8},
    },
    "shapley": {
        "amoeneburg": {
            "subsize_frac": 0.80,
            "convergence_error": 0.10,
            "learning_rate": 1.000000e-06,
            "max_iteration": 30,
        },
        "maerker": {
            "subsize_frac": 0.7,
            "convergence_error": 0.1,
            "learning_rate": 1.000000e-06,
            "max_iteration": 30,
        },
        "rohrdorfer": {
            "subsize_frac": 0.8,
            "convergence_error": 0.1,
            "learning_rate": 1.000000e-06,
            "max_iteration": 30,
        },
        "spenner": {
            "subsize_frac": 0.7,
            "convergence_error": 0.10,
            "learning_rate": 1.000000e-07,
            "max_iteration": 40,
        },
        "woessingen": {
            "subsize_frac": 0.7,
            "convergence_error": 0.05,
            "learning_rate": 1.000000e-06,
            "max_iteration": 30,
        },
    },
    "lof": {
        "amoeneburg": {"nr_of_neighbors": 3},
        "maerker": {"nr_of_neighbors": 8},
        "rohrdorfer": {"nr_of_neighbors": 4},
        "spenner": {"nr_of_neighbors": 5},
        "woessingen": {"nr_of_neighbors": 6},
    },
    "full": {
        "amoeneburg": {},
        "maerker": {},
        "rohrdorfer": {},
        "spenner": {},
        "woessingen": {},
    },
    "fixed_window": {
        "amoeneburg": {"subsize_frac": 0.85},
        "maerker": {"subsize_frac": 0.7},
        "rohrdorfer": {"subsize_frac": 0.65},
        "spenner": {"subsize_frac": 0.75},
        "woessingen": {"subsize_frac": 0.6},
    },
    "mutual_information": {
        "amoeneburg": {"alpha": 0.05, "nr_of_neighbors": 3},
        "maerker": {"alpha": 0.05, "nr_of_neighbors": 5},
        "rohrdorfer": {"alpha": 0.01, "nr_of_neighbors": 3},
        "spenner": {"alpha": 0.05, "nr_of_neighbors": 5},
        "woessingen": {"alpha": 0.05, "nr_of_neighbors": 5},
    },
    "fish": {
        "amoeneburg": {
            "neighborhood_size": 7,
            "spatial_weight": 1,
            "temporal_weight": 0.5,
        },
        "maerker": {
            "neighborhood_size": 7,
            "spatial_weight": 1,
            "temporal_weight": 0.5,
        },
        "rohrdorfer": {
            "neighborhood_size": 7,
            "spatial_weight": 1,
            "temporal_weight": 0.5,
        },
        "spenner": {
            "neighborhood_size": 7,
            "spatial_weight": 1,
            "temporal_weight": 0.5,
        },
        "woessingen": {
            "neighborhood_size": 7,
            "spatial_weight": 1,
            "temporal_weight": 0.5,
        },
    },
    "fish1": {
        "amoeneburg": {
            "subsize_frac": 0.6,
            "temporal_weight": 0.7,
        },
        "maerker": {
            "subsize_frac": 0.7,
            "temporal_weight": 0.7,
        },
        "rohrdorfer": {
            "subsize_frac": 0.9,
            "temporal_weight": 0.9,
        },
        "spenner": {
            "subsize_frac": 0.9,
            "temporal_weight": 0.1,
        },
        "woessingen": {
            "subsize_frac": 0.6,
            "temporal_weight": 0.8,
        },
    },
    "random": {
        "amoeneburg": {"subsize_frac": 1},
        "maerker": {"subsize_frac": 1},
        "rohrdorfer": {"subsize_frac": 1},
        "spenner": {"subsize_frac": 1},
        "woessingen": {"subsize_frac": 1},
    },
}
