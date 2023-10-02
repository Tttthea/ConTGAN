import pm4py
import pandas as pd
pd.set_option('display.max_columns', None)
if __name__ == "__main__":
    log = pm4py.read_xes('dataset/Sepsis.xes')
    df = pm4py.convert_to_dataframe(log)

    print(df)

    # net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log)
    # pm4py.view_petri_net(net, initial_marking, final_marking, format="svg")