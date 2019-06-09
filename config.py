class TargetProgram:
    def __init__(self, progname, progdir):
        self.progname = progname
        self.progdir = progdir
        self.output = self.progdir + '/output'
        self.output2 = self.progdir + '/output2'

        self.static_ins_filename = self.output + "/static_instructions.out"
        self.dynamic_ins_filename = self.output + "/dynamic_instruction.out"
        self.memory_rw_filename = self.output + "/memory_rw_address.out"
        self.prop_his_filename = self.output + "/prop_his.out"
        self.fi_point_filename = self.output + '/fi_point_opt.out'
        self.detected_fi_point_filename = self.output + "/reliability_analysis/detected/detected_fi_point_opt.out"
        self.unmasked_fi_point_filename = self.output + "/reliability_analysis/unmasked/unmasked_fi_point_opt.out"
        self.unmasked_fi_point_filename_from_basic_error_set = self.output + "/reliability_analysis/unmasked_from_basic_error_set/self.unmasked_from_basic_error_set_fi_point_opt.out"
        self.unmasked_undetected_fi_point_filename = self.output + "/reliability_analysis/unmasked_undetected/unmasked_undetected_fi_point_opt.out"
        self.no_effect_fi_point_filename = self.output + "/reliability_analysis/no_effect/no_effect_fi_point_opt.out"
        self.unmasked_undetected_fi_point_filename_precise = self.output + "/reliability_analysis/unmasked_undetected_precise/self.unmasked_undetected_precise_fi_point_opt_precise.out"
        self.dot_file_name = self.output + "/reliability_analysis/dots/mxk-gen.cds.dot"
        self.png_file_name = self.output + "/reliability_analysis/pngs/mxk-gen.cds.png"

        self.ins_infos_filename = self.output2 + '/ins_infos.csv'
        self.dataflow_similarity_features_filename = self.output2 + '/dataflow_similarity_features.csv'
        self.dataflow_similarity_result = self.output2 + '/dataflow_similarity_result.csv'
        self.dataflow_similarity_fi_point_filename = self.output2 + '/dataflow_similarity_fi_point.out'

        self.dynamic_ins_filename2 = self.output2 + '/dynamic_instruction.csv'
        self.block_id_ins_filename = self.output2 + '/block_id_ins.csv'

        self.machine_states_path = self.progdir + "/output/machine_states.out"
        self.results_path = self.progdir + "/output/results.out"
        self.full_space_results_path = self.progdir + "/output/full_space_results.csv"
        self.results_with_machine_states_path = self.progdir + "/fault_similarity_info/results_with_machine_states.csv"
        self.full_space_results_with_machine_states_path = self.progdir + "/fault_similarity_info/full_space_results_with_machine_states.csv"
        self.machine_states_with_prop_his_with_results_path = self.progdir + "/fault_similarity_info/machine_states_with_prop_his_with_results.csv"

        self.df_raw_data_remove_trivial_path = self.progdir + '/fault_similarity_info/df_raw_data_remove_trivial.csv'
        self.full_space_csv_file_name = self.progdir + '/fault_similarity_info/full_space_csv.csv'
        self.prop_his_path = self.progdir + "/fault_similarity_info/prop_his.out"
        self.raw_data_path = "newresult.csv"  # default "data/usdchf-mt4.csv"


class Config:
    run_mode = "TEST"  # TRAIN TEST TRAIN_TEST
    full_space_validation = True
    # data
    need_remerge_tables = True
    using_prop_his = False
    using_selected_features = True
    using_dataflow = True
    max_size = 10000
    full_space_max_size = 10000000

    test_size = 0.5

    fake_scale = False

    # FEATURES = ['dyn', 'ip_x', 'REG_RDI', 'REG_RSI', 'REG_RBP', 'REG_RSP', 'REG_RBX', 'REG_RDX', 'REG_RCX', 'REG_RAX',
    #             'REG_SEG_CS', 'REG_SEG_SS', 'REG_SEG_DS', 'REG_SEG_ES', 'REG_SEG_FS', 'REG_SEG_GS', 'REG_RFLAGS',
    #             'REG_RIP', 'memread_addr_1', 'read_size_1', 'mem_val_1', 'displacement_1', 'base_reg_1', 'index_reg_1',
    #             'scale_1', 'memread_addr_2', 'read_size_2', 'mem_val_2', 'displacement_2', 'base_reg_2', 'index_reg_2',
    #             'scale_2', 'memwrite_addr', 'read_size', 'mem_val', 'displacement', 'base_reg', 'index_reg', 'scale',
    #             'blockid', 'prop_his', 'num', 'ip_y', 'reg', 'bit', 'left', 'right', 'result']
    # if using_prop_his:
    #     # FEATURES = ['dyn', 'ip_x', 'REG_RDI', 'REG_RSI', 'REG_RBP', 'REG_RSP', 'REG_RBX', 'REG_RDX', 'REG_RCX', 'REG_RAX',
    #     #             'REG_SEG_CS', 'REG_SEG_SS', 'REG_SEG_DS', 'REG_SEG_ES', 'REG_SEG_FS', 'REG_SEG_GS', 'REG_RFLAGS',
    #     #             'REG_RIP', 'memread_addr_1', 'read_size_1', 'mem_val_1', 'displacement_1', 'base_reg_1', 'index_reg_1',
    #     #             'scale_1', 'memread_addr_2', 'read_size_2', 'mem_val_2', 'displacement_2', 'base_reg_2', 'index_reg_2',
    #     #             'scale_2', 'memwrite_addr', 'read_size', 'mem_val', 'displacement', 'base_reg', 'index_reg', 'scale',
    #     #             'blockid', 'prop_his', 'num', 'ip_y', 'reg', 'bit', 'left', 'right', 'result']
    #
    #     FEATURES = ['dyn', 'ip_x', 'REG_RDI', 'REG_RSI', 'REG_RBP', 'REG_RSP', 'REG_RBX', 'REG_RDX', 'REG_RCX',
    #                 'REG_RAX', 'reg', 'bit',
    #                 'blockid', 'prop_his', 'result']

    if using_dataflow:

        FEATURES = ['dyn', 'ip_x', 'reg', 'REG_RDI', 'REG_RSI', 'REG_RBP', 'REG_RSP', 'REG_RBX', 'REG_RDX', 'REG_RCX',
                    'REG_RAX', 'bit', 'length', 'result']
    else:
        # FEATURES = ['dyn', 'ip_x', 'REG_RDI', 'REG_RSI', 'REG_RBP', 'REG_RSP', 'REG_RBX', 'REG_RDX', 'REG_RCX', 'REG_RAX',
        #             'REG_SEG_CS', 'REG_SEG_SS', 'REG_SEG_DS', 'REG_SEG_ES', 'REG_SEG_FS', 'REG_SEG_GS', 'REG_RFLAGS',
        #             'REG_RIP', 'memread_addr_1', 'read_size_1', 'mem_val_1', 'displacement_1', 'base_reg_1', 'index_reg_1',
        #             'scale_1', 'memread_addr_2', 'read_size_2', 'mem_val_2', 'displacement_2', 'base_reg_2', 'index_reg_2',
        #             'scale_2', 'memwrite_addr', 'read_size', 'mem_val', 'displacement', 'base_reg', 'index_reg', 'scale',
        #             'num', 'ip_y', 'reg', 'bit', 'left', 'right', 'result']
        FEATURES = ['ip_x', 'REG_RDI', 'REG_RSI', 'REG_RBP', 'REG_RSP', 'REG_RBX', 'REG_RDX', 'REG_RCX', 'REG_RAX',
                    'REG_SEG_CS', 'REG_SEG_SS', 'REG_SEG_DS', 'REG_SEG_ES', 'REG_SEG_FS', 'REG_SEG_GS', 'REG_RFLAGS',
                    'REG_RIP', 'bit', 'reg', 'length', 'result']
        # FEATURES = ['dyn', 'ip_x', 'REG_RDI', 'REG_RSI', 'REG_RBP', 'REG_RSP', 'REG_RBX', 'REG_RDX', 'REG_RCX',
        #             'REG_RFLAGS',
        #             'REG_RAX', 'reg', 'bit',
        #             'result']
        # 'result', 'length']

    # FEATURES = ['dyn', 'ip_x', 'REG_RSP', 'REG_RBP', 'REG_RSP', 'REG_RBX', 'REG_RDX', 'REG_RCX', 'REG_RAX', 'blockid', 'prop_his', 'result']
    # FEATURES = ['dyn', 'ip_x', 'REG_RSP', 'REG_RBP', 'reg', 'result']

    LABELS = ['result']  # if prediction_mode = 1, default ['label'], if prediction_mode = 2, default ['ri']
    BLOCK_ID_LABEL = 'blockid'
    HISTORY_ID_LABEL = 'prop_his'
    COLUMS = FEATURES + LABELS
    feature_dim = len(FEATURES)  # How many features in the data

    # model
    selection_algorithm = "NSGA2"  # NSGA2 SPEA2
    mutate_type = "RANDOM_NUMBER"  # BITWISE ALLBITS 8BITS RANDOM_NUMBER
    init_type = "ALLZERO"  # ALLZERO RANDOM ALLONE
    model = "SINGLE_PROJ"  # MULTI_PROJ SINGLE_PROJ
    indpb = 0.3
    NGEN = 100
    MU = 100
    CXPB = 0.5
    epoch_size = 10

    # model save and restore
    restore_model = True  # is restore model from model_save_path
    retrain_model = False
    model_save_path = "save/model/model.ckpt"

    # evaluation
    # eval_mse, eval_exclusive, eval_force_merge_prop_his, eval_sdc_only, normal, def_use_length, length_same_reg, with_dataflow
    evaluation_method = "with_dataflow"
    exclusive_base = 0.6
    min_sub_block_size = 1000

    version = "190528-0.002"

    # others
    base_dir = "/home/xiaofengwo/sdb1/fault_injection"

    Benchmarks = [
        TargetProgram(
            progname='basicmath',
            progdir=base_dir + '/fi_target/ch2_benchmarks/basicmath',
        ),

        TargetProgram(  # program output is not same between each running, but these codes have been removed.
            progname='bitcount',
            progdir=base_dir + '/fi_target/ch2_benchmarks/bitcount',
        ),
        TargetProgram(
            progname='qsort',
            progdir=base_dir + '/fi_target/ch2_benchmarks/qsort',
        ),
        TargetProgram(
            progname='susan',
            progdir=base_dir + '/fi_target/ch2_benchmarks/susan',
        ),

        TargetProgram(
            progname='blowfish',
            progdir=base_dir + '/fi_target/ch2_benchmarks/blowfish',
        ),

        TargetProgram(
            progname='dijkstra',
            progdir=base_dir + '/fi_target/ch2_benchmarks/dijkstra',
        ),
        TargetProgram(
            progname='patricia',
            progdir=base_dir + '/fi_target/ch2_benchmarks/patricia',
        ),
        TargetProgram(
            progname='FFT',
            progdir=base_dir + '/fi_target/ch2_benchmarks/FFT',
        )
    ]

    current_program = Benchmarks[0]
