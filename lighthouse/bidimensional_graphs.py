import os
import argparse

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='2D-plots',
        description='Generate 2D interactive graphs of LLM inference performance.',
    )
    parser.add_argument('-x', '--xaxes', action='store', default='GPU Layers', choices=['GPU Layers', 'Decode Threads', 'Prefill Threads', 'Prompt Length', 'Num. Requests'], type=str, help="X axes parameter.")
    parser.add_argument('--run-anchor', action='store', nargs='+', type=str, required=True, help="Anchor value for run name.")
    parser.add_argument('--store', action='store_true', default=False, help="Store graphs in images folder.")

    return parser.parse_args()

def main():

    args = parse_arguments()

    df = pd.read_csv('bulb.csv').drop(['id'], axis=1)

     # Create a subplot figure with 1 row and 2 columns
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for run_anchor in args.run_anchor:        
        filtered_df = df[(df['Run Name']==run_anchor)]

        fig.add_trace(
            go.Scatter(
                x=filtered_df[args.xaxes],
                y=filtered_df["Prefill Time (tk/s)"],
                hovertemplate='%{y:.2f}',
                mode='markers',
                name=f'{run_anchor}'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=filtered_df[args.xaxes],
                y=filtered_df["Decode Time (tk/s)"],
                hovertemplate='%{y:.2f}',
                mode='markers',
                name=f'{run_anchor}'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=filtered_df[args.xaxes],
                y=filtered_df["Total Time (s)"],
                customdata=filtered_df[["Load Time (s)", "Prefill Time (s)", "Decode Time (s)"]].to_numpy(),
                hovertemplate='<b>Total Time: %{y:.2f}</b> <br>Load Time: %{customdata[0]:.1f} <br>Prefill Time: %{customdata[2]:.1f} <br>Decode Time: %{customdata[3]:.1f}',
                mode='markers',
                name=f'{run_anchor}'),
            row=3, col=1
        )

    fig.update_yaxes(title_text="Prefill Time (tk/s)", row=1, col=1)
    fig.update_yaxes(title_text="Decode Time (tk/s)", row=2, col=1)
    fig.update_yaxes(title_text="Latency (s)", row=3, col=1)
    fig.update_xaxes(title_text=args.xaxes, row=3, col=1)

    # Update layout if needed
    fig.update_layout(height=700, width=500, showlegend=False)
    fig_name = '_'.join([args.xaxes, '_'.join(set(df[df["Run Name"].isin(args.run_anchor)]["Device"])), '_'.join(set(df[df["Run Name"].isin(args.run_anchor)]["Model"]))])

    # Show the figure
    fig.show()
    if args.store:
        fig.write_image(os.path.join('images', f'{fig_name}.png'))
        fig.write_html(os.path.join('images', f'{fig_name}.html'))


if __name__ == "__main__":
    main()