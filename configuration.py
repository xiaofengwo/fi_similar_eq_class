class Config:

    with_prop_his = True
    mutate_type = "BITWISE"  # BITWISE ALLBITS

    max_size = 1000
    test_size = 0.50
    # train_batch_size = 10
    # test_batch_size = 1
    indpb = 0.01

    NGEN = 1000
    MU = 100
    CXPB = 0.9
    epoch_size = 10

    # machine_states_path = "basicmath/machine_states.out"
    # results_path = "basicmath/results.out"
    # results_with_machine_states_path = "basicmath/results_with_machine_states_path.csv"

    # machine_states_path = "data/basicmath2e5his/machine_states.out"
    # results_path = "data/basicmath2e5his/results.out"
    # results_with_machine_states_path = "data/basicmath2e5his/results_with_machine_states_path.csv"
    # machine_states_with_prop_his_with_results_path = "data/basicmath2e5his/machine_states_with_prop_his_with_results.csv"
    # prop_his_path = 'data/basicmath2e5his/prop_his.out'

    # machine_states_path = "data/basicmath1e4his/machine_states.out"
    # results_path = "data/basicmath1e4his/results.out"
    # results_with_machine_states_path = "data/basicmath1e4his/results_with_machine_states_path.csv"
    # machine_states_with_prop_his_with_results_path = "data/basicmath1e4his/machine_states_with_prop_his_with_results.csv"
    # prop_his_path = 'data/basicmath1e4his/prop_his.out'

    machine_states_path = "/home/xiaofengwo/fault_injection/output/susan/machine_states.out"
    results_path = "/home/xiaofengwo/fault_injection/output/susan/results.out"
    results_with_machine_states_path = "/home/xiaofengwo/fault_injection/output/susan/results_with_machine_states_path.csv"
    machine_states_with_prop_his_with_results_path = "/home/xiaofengwo/fault_injection/output/susan/machine_states_with_prop_his_with_results.csv"
    prop_his_path = '/home/xiaofengwo/fault_injection/output/susan/prop_his.out'


    raw_data_path = "newresult.csv"       # default "data/usdchf-mt4.csv"
    version = "170725-0.003"
    # model save and restore
    model_save_path = "save/model/model.ckpt"
    restore_model = False                       # is restore model from model_save_path

    FEATURES = ['dyn', 'ip', 'eax', 'ecx', '[i]', '[a]', '[b]', '[sum]', '[d]', '[b]']       # default ['open', 'high', 'low', 'close', 'volume']
    LABELS = ['result']                           # if prediction_mode = 1, default ['label'], if prediction_mode = 2, default ['ri']
    COLUMS = FEATURES + LABELS
    feature_dim = len(FEATURES)                  # How many features in the data
