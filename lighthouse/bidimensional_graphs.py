import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='2D-plots',
        description='Generate 2D interactive graphs of LLM inference performance.',
    )
    parser.add_argument('-x', '--xaxes', action='store', default='GPU Layers', choices=['GPU Layers', 'Threads', 'Batch Threads'], type=str, help="x axes parameter")
    parser.add_argument('--threads_anchor', action='store', default=-1, type=int, help="anchor value for number of threads")
    parser.add_argument('--batch_threads_anchor', action='store', default=-1, type=int, help="anchor value for number of batch threads")
    parser.add_argument('--gpu_anchor', action='store', default=-1, type=float, help="anchor value for number of threads")

    
    return parser.parse_args()

def main():

    args = parse_arguments()

    df = pd.read_csv('output/bulb.csv').drop(['id'], axis=1)
    threads_filter = df['Threads'] == args.threads_anchor if args.threads_anchor!=-1 else df['Threads'] == df['Threads']
    batch_threads_filter = df['Batch Threads'] == args.batch_threads_anchor if args.batch_threads_anchor!=-1 else df['Batch Threads'] == df['Batch Threads']
    gpu_filter = df['GPU Layers'] == args.gpu_anchor if args.gpu_anchor!=-1 else df['GPU Layers'] == df['GPU Layers']
    
    filtered_df = df[(df['Node ID']=='Quadro P1000-3.94-15.35-12-llama-2-7b-chat.Q2_K.gguf') & 
                    threads_filter &
                    batch_threads_filter &
                    gpu_filter
                    ]

    # Create a subplot figure with 1 row and 2 columns
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
    fig.add_trace(
        go.Scatter(x=filtered_df[args.xaxes], y=filtered_df["Prompt Eval Time (Tk/s)"], mode='markers', name=''),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=filtered_df[args.xaxes], y=filtered_df["Eval Time (Tk/s)"], mode='markers', name=''),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=filtered_df[args.xaxes], y=filtered_df["Total Time (s)"], mode='markers', name=''),
        row=3, col=1
    )

    fig.update_yaxes(title_text="Prompt Eval Time (Tk/s)", row=1, col=1)
    fig.update_yaxes(title_text="Eval Time (Tk/s)", row=2, col=1)
    fig.update_yaxes(title_text="Total Time (s)", row=3, col=1)
    fig.update_xaxes(title_text=args.xaxes, row=3, col=1)

    # Update layout if needed
    fig.update_layout(height=700, width=500, showlegend=False)

    # Show the figure
    fig.show()

if __name__ == "__main__":
    main()
