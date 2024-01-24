import os
import argparse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 'Quadro P1000-3.94-15.35-12-llama-2-7b-chat.Q2_K.gguf'

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='2D-plots',
        description='Generate 2D interactive graphs of LLM inference performance.',
    )
    parser.add_argument('-x', '--xaxes', action='store', default='GPU Layers', choices=['GPU Layers', 'Threads', 'Batch Threads', 'Prompt Length'], type=str, help="x axes parameter")
    parser.add_argument('--run_anchor', action='store', nargs='+', type=str, required=True, help="anchor value for run name")
    # parser.add_argument('--node_anchor', action='store', type=str, required=True, help="anchor value for current machine and model")
    # parser.add_argument('--threads_anchor', action='store', default=-1, type=int, help="anchor value for number of threads")
    # parser.add_argument('--batch_threads_anchor', action='store', default=-1, type=int, help="anchor value for number of batch threads")
    # parser.add_argument('--gpu_anchor', action='store', default=-1, type=float, help="anchor value for number of threads")

    return parser.parse_args()

def main():

    args = parse_arguments()

    df = pd.read_csv('output/bulb.csv').drop(['id'], axis=1)

     # Create a subplot figure with 1 row and 2 columns
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    # threads_filter = df['Threads'] == args.threads_anchor if args.threads_anchor!=-1 else df['Threads'] == df['Threads']
    # batch_threads_filter = df['Batch Threads'] == args.batch_threads_anchor if args.batch_threads_anchor!=-1 else df['Batch Threads'] == df['Batch Threads']
    # gpu_filter = df['GPU Layers'] == args.gpu_anchor if args.gpu_anchor!=-1 else df['GPU Layers'] == df['GPU Layers']

    # filtered_df = df[(df['Node ID']==args.node_anchor) & 
    #                 threads_filter &
    #                 batch_threads_filter &
    #                 gpu_filter
    #                 ]

    for run_anchor in args.run_anchor:        
        filtered_df = df[(df['Run Name']==run_anchor)]

        fig.add_trace(
            go.Scatter(
                x=filtered_df[args.xaxes],
                y=filtered_df["Prompt Eval Time (Tk/s)"],
                hovertemplate='%{y:.2f}',
                mode='markers',
                name=f'{run_anchor}'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=filtered_df[args.xaxes],
                y=filtered_df["Eval Time (Tk/s)"],
                hovertemplate='%{y:.2f}',
                mode='markers',
                name=f'{run_anchor}'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=filtered_df[args.xaxes],
                y=filtered_df["Total Time (s)"],
                customdata=filtered_df[["Load Time (s)", "Sample Time (s)", "Prompt Eval Time (s)", "Eval Time (s)"]].to_numpy(),
                hovertemplate='<b>Total Time: %{y:.2f}</b> <br>Load Time: %{customdata[0]:.1f} <br>Sample Time: %{customdata[1]:.2f} <br>Prompt Eval Time: %{customdata[2]:.1f} <br>Eval Time: %{customdata[3]:.1f}',
                mode='markers',
                name=f'{run_anchor}'),
            row=3, col=1
        )

    fig.update_yaxes(title_text="Prompt Eval Time (Tk/s)", row=1, col=1)
    fig.update_yaxes(title_text="Eval Time (Tk/s)", row=2, col=1)
    fig.update_yaxes(title_text="Latency (s)", row=3, col=1)
    fig.update_xaxes(title_text=args.xaxes, row=3, col=1)

    # Update layout if needed
    fig.update_layout(height=700, width=500, showlegend=False)
    fig_name = '_'.join([args.xaxes, '_'.join(set(df[df["Run Name"].isin(args.run_anchor)]["Device"])), '_'.join(set(df[df["Run Name"].isin(args.run_anchor)]["Model"]))])

    # Show the figure
    fig.show()
    fig.write_image(os.path.join('images', f'{fig_name}.png'))
    fig.write_html(os.path.join('images', f'{fig_name}.html'))


if __name__ == "__main__":
    main()