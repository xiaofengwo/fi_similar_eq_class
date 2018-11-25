class Config:
    test_size = 0.99
    # train_batch_size = 10
    # test_batch_size = 1

    # machine_states_path = "basicmath/machine_states.out"
    # results_path = "basicmath/results.out"
    # results_with_machine_states_path = "basicmath/results_with_machine_states_path.csv"

    machine_states_path = "data/basicmath100000/machine_states.out"
    results_path = "data/basicmath100000/results.out"
    results_with_machine_states_path = "data/basicmath100000/results_with_machine_states_path.csv"

    raw_data_path = "newresult.csv"       # default "data/usdchf-mt4.csv"
    version = "170725-0.003"
    # model save and restore
    model_save_path = "save/model/model.ckpt"
    restore_model = True                       # is restore model from model_save_path

    FEATURES = ['dyn', 'ip', 'eax', 'ecx', '[i]', '[a]', '[b]', '[sum]', '[d]', '[b]']       # default ['open', 'high', 'low', 'close', 'volume']
    LABELS = ['result']                           # if prediction_mode = 1, default ['label'], if prediction_mode = 2, default ['ri']
    COLUMS = FEATURES + LABELS
    feature_dim = len(FEATURES)                  # How many features in the data
