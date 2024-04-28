#prototype for plots of training, validation and test loss
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import Constants as const

def plot_mean_std(path_list):
    fig = go.Figure()

    # Define a light grey color for the shaded area
    light_grey_color = 'rgba(200, 200, 200, 0.5)' # 50% opacity for a lighter shade

    for path in path_list:
        df = pd.read_csv(path)
        name = path.stem.split('_')[0] + '_' + path.stem.split('_')[1]

        # Calculate adjusted lower bound (non-negative)
        lower_bound = np.maximum(df['mean'] - df['std'], 0)

        # Add mean line trace
        fig.add_trace(go.Scatter(x=df.index, y=df['mean'], mode='lines', name=name + ' mean'))

        # Add dashed lines for the boundaries of standard deviation
        fig.add_trace(go.Scatter(x=df.index, y=df['mean'] + df['std'], mode='lines', name=name + ' +std', line=dict(dash='dash'), visible='legendonly'))
        fig.add_trace(go.Scatter(x=df.index, y=lower_bound, fill='tonexty', mode='lines', line=dict(width=0), fillcolor=light_grey_color, name=name + ' std area', visible='legendonly'))
        fig.add_trace(go.Scatter(x=df.index, y=lower_bound, mode='lines', name=name + ' -std', line=dict(dash='dash'), visible='legendonly'))

    # preselect shown area on y-axis
    fig.update_yaxes(range=[0, 2*df['mean'].mean()])
    # Add button to switch between log and linear y-axis
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
                buttons=[
                    dict(
                        label="Log Scale",
                        method="relayout",
                        args=[{"yaxis.type": "log"}]
                    ),
                    dict(
                        label="Linear Scale",
                        method="relayout",
                        args=[{"yaxis.type": "linear"}]
                    )
                ]
            )
        ]
    )

    return fig
path_res=Path(const.RESULTS_PATH,'Experiment_2_training_behavior_4th_order_RLC')
gen_res_files=False

if gen_res_files:
    name_models=['WhiteBoxODE','GreyBoxODE1','GreyBoxODE2','GreyBoxODE3','BlackBoxODE','BlackBoxLSTM']
    seeds=[0,42,123,205]
    results_mean_std = {}
    #collect results for all seeds
    results={model : {} for model in name_models}
    for name_model in name_models:
        results[name_model] = {seed: {} for seed in seeds}
        for seed in seeds:
            results[name_model][seed]= pd.read_csv(Path(path_res, f'{name_model}_seed_{seed}','results_training.csv'), index_col=0)

    for name_model in name_models:
        results_mean_std[name_model] = {}

        #aggregate the all dataframes for each model
        for key in results[name_model][seeds[0]].keys():
            results_mean_std[name_model][key] = pd.concat([results[name_model][seed][key] for seed in seeds]).groupby(level=0).agg(['mean', 'std'])

    # save the results
    for name_model in name_models:
        for key in results_mean_std[name_model].keys():
            results_mean_std[name_model][key].to_csv(Path(path_res, f'{name_model}_{key}_mean_std.csv'))





path_WhiteBoxODE_train=Path(path_res,'WhiteBoxODE_train_mean_std.csv')
path_WhiteBoxODE_val=Path(path_res,'WhiteBoxODE_val_mean_std.csv')
path_WhiteBoxODE_test=Path(path_res,'WhiteBoxODE_test_mean_std.csv')


path_list_knownODE=[path_WhiteBoxODE_train,path_WhiteBoxODE_val,path_WhiteBoxODE_test]

fig=plot_mean_std(path_list_knownODE)
fig.update(layout_title_text='WhiteBoxODE')
#save figure
fig.write_html(Path(path_res,"WhiteBoxODE_loss.html"))


path_GreyBoxODE1_train=Path(path_res,'GreyBoxODE1_train_mean_std.csv')
path_GreyBoxODE1_val=Path(path_res,'GreyBoxODE1_val_mean_std.csv')
path_GreyBoxODE1_test=Path(path_res,'GreyBoxODE1_test_mean_std.csv')

path_list_GreyBoxODE1=[path_GreyBoxODE1_train,path_GreyBoxODE1_val,path_GreyBoxODE1_test]
fig=plot_mean_std(path_list_GreyBoxODE1)
fig.update(layout_title_text='GreyBoxODE1')
#save figure
fig.write_html(Path(path_res,"GreyBoxODE1_loss.html"))

path_GreyBoxODE2_train=Path(path_res,'GreyBoxODE2_train_mean_std.csv')
path_GreyBoxODE2_val=Path(path_res,'GreyBoxODE2_val_mean_std.csv')
path_GreyBoxODE2_test=Path(path_res,'GreyBoxODE2_test_mean_std.csv')

path_list_GreyBoxODE2=[path_GreyBoxODE2_train,path_GreyBoxODE2_val,path_GreyBoxODE2_test]
fig=plot_mean_std(path_list_GreyBoxODE2)
fig.update(layout_title_text='GreyBoxODE2')
#save figure
fig.write_html(Path(path_res,"GreyBoxODE2_loss.html"))

#greybox model 3
path_GreyBoxODE3_train=Path(path_res,'GreyBoxODE3_train_mean_std.csv')
path_GreyBoxODE3_val=Path(path_res,'GreyBoxODE3_val_mean_std.csv')
path_GreyBoxODE3_test=Path(path_res,'GreyBoxODE3_test_mean_std.csv')

path_list_GreyBoxODE3=[path_GreyBoxODE3_train,path_GreyBoxODE3_val,path_GreyBoxODE3_test]
fig=plot_mean_std(path_list_GreyBoxODE3)
fig.update(layout_title_text='GreyBoxODE3')
#save figure
fig.write_html(Path(path_res,"GreyBoxODE3_loss.html"))



#black box ODE
path_BlackBoxODE_train=Path(path_res,'BlackBoxODE_train_mean_std.csv')
path_BlackBoxODE_val=Path(path_res,'BlackBoxODE_val_mean_std.csv')
path_BlackBoxODE_test=Path(path_res,'BlackBoxODE_test_mean_std.csv')


path_list_BlackBoxODE=[path_BlackBoxODE_train,path_BlackBoxODE_val,path_BlackBoxODE_test]
fig=plot_mean_std(path_list_BlackBoxODE)
fig.update(layout_title_text='BlackBoxODE')
#save figure
fig.write_html(Path(path_res,"BlackBoxODE_loss.html"))

#black box LSTM
path_BlackBoxLSTM_train=Path(path_res,'BlackBoxLSTM_train_mean_std.csv')
path_BlackBoxLSTM_val=Path(path_res,'BlackBoxLSTM_val_mean_std.csv')
path_BlackBoxLSTM_test=Path(path_res,'BlackBoxLSTM_test_mean_std.csv')

path_list_BlackBoxLSTM=[path_BlackBoxLSTM_train,path_BlackBoxLSTM_val,path_BlackBoxLSTM_test]
fig=plot_mean_std(path_list_BlackBoxLSTM)
fig.update(layout_title_text='BlackBoxLSTM')
#save figure
fig.write_html(Path(path_res,"BlackBoxLSTM_loss.html"))

#all in one
fig=plot_mean_std(path_list_knownODE+path_list_GreyBoxODE1+path_list_GreyBoxODE2+path_list_GreyBoxODE3+path_list_BlackBoxODE+path_list_BlackBoxLSTM)
fig.update(layout_title_text='All')
#save figure
fig.write_html(Path(path_res,"All_loss.html"))

#all training
paths_train=[path_WhiteBoxODE_train,path_GreyBoxODE1_train,path_GreyBoxODE2_train,path_GreyBoxODE3_train,path_BlackBoxODE_train,path_BlackBoxLSTM_train]
fig=plot_mean_std(paths_train)
fig.update(layout_title_text='All training')
#save figure
fig.write_html(Path(path_res,"All_training_loss.html"))

#all validation
paths_val=[path_WhiteBoxODE_val,path_GreyBoxODE1_val,path_GreyBoxODE2_val,path_GreyBoxODE3_val,path_BlackBoxODE_val,path_BlackBoxLSTM_val]
fig=plot_mean_std(paths_val)
fig.update(layout_title_text='All validation')
#save figure
fig.write_html(Path(path_res,"All_validation_loss.html"))

#all test
paths_test=[path_WhiteBoxODE_test,path_GreyBoxODE1_test,path_GreyBoxODE2_test,path_GreyBoxODE3_test,path_BlackBoxODE_test,path_BlackBoxLSTM_test]
fig=plot_mean_std(paths_test)
fig.update(layout_title_text='All test')
#save figure
fig.write_html(Path(path_res,"All_test_loss.html"))







