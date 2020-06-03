import numpy as np
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc

from skimage import io, transform, filters, segmentation
from skimage import img_as_ubyte
from scipy import ndimage
import plotly.express as px
import plotly.graph_objects as go
from dash_canvas.utils import array_to_data_url


def make_figure(img_array):
    img_uri = array_to_data_url(img_array)
    width = img_array.shape[1]
    height = img_array.shape[0]
    fig = go.Figure()
    # Add trace
    fig.add_trace(
        go.Scatter(x=[], y=[])
    )
    # Add images
    fig.add_layout_image(
        dict(
            source=img_uri,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=width,
            sizey=height,
            sizing="contain",
            layer="below"
        )
    )
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, range=(0, width),
    showticklabels=False,
    zeroline=False)
    fig.update_yaxes(showgrid=False, scaleanchor='x', range=(height, 0),
    showticklabels=False,
    zeroline=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def _crop_to_shape(img, layout_shape):
    if layout_shape is None:
        return img
    else:
        return img[int(layout_shape['y0']):int(layout_shape['y1']),
                   int(layout_shape['x0']):int(layout_shape['x1'])]


img = io.imread('assets/uEye_Image_000827.png')
img = img_as_ubyte(transform.rescale(img, 0.5))

fig = make_figure(img)
fig.update_layout(dragmode='drawrect')

app = dash.Dash(__name__)
server = app.server


app.layout = html.Div(children=[
    html.Div([
        dcc.Graph(id='graph', figure=fig),
        dcc.Store(id='store'),
        dcc.Store(id='store-contour'),
        ], style={'width':'45%'}),
    html.Div([
        html.Button('Crop to rectangle', id='crop-button'),
        html.Button('Back to original shape', id='back-button'),
        html.Button('Find contour', id='contour-button'),
        html.H3('Model parameters'),
        html.H5('Parameter 1'),
        dcc.Input(id='input-param1', type='number', value=0)
        ], style={'width':'45%'}),
        ])

@app.callback(
     dash.dependencies.Output('store', 'data'),
    [dash.dependencies.Input('graph', 'relayoutData')])
def store_rectangle(fig_data):
    if fig_data is not None and 'shapes' in fig_data:
        return fig_data['shapes'][-1]
    else:
        return dash.no_update

@app.callback(
     [dash.dependencies.Output('graph', 'figure'),
      dash.dependencies.Output('store-contour', 'data')],
    [dash.dependencies.Input('crop-button', 'n_clicks'),
     dash.dependencies.Input('back-button', 'n_clicks'),
     dash.dependencies.Input('contour-button', 'n_clicks')
     ],
    [dash.dependencies.State('store', 'data')])
def update_figure(click_crop, click_back, click_contour, shape_data):
    if click_back is None and click_crop is None:
        return dash.no_update, dash.no_update
    else:
        ctx = dash.callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        print(button_id)
        contour = None
        if button_id == 'crop-button':
            img_crop = _crop_to_shape(img, shape_data)
            new_store_data = dash.no_update
        elif button_id == 'back-button':
            img_crop = img
        else: # contour-button
            img_crop = _crop_to_shape(img, shape_data)
            threshold = filters.threshold_otsu(img_crop)
            mask = img_crop < threshold
            mask = ndimage.binary_fill_holes(mask)
            img_crop = img_as_ubyte(segmentation.mark_boundaries(img_crop, mask, mode='thick'))
            contour = np.nonzero(mask)
        fig = make_figure(img_crop)
        fig.update_layout(dragmode='drawrect')
        return fig, contour


if __name__ == '__main__':
    app.run_server(debug=True)

