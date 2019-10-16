import os, sys, re, time
import numpy as np
from functools import reduce
from itertools import cycle
from bokeh.models import HoverTool, ColumnDataSource, TapTool
from bokeh.plotting import figure, show, output_file
from bokeh.models.widgets import Toggle, Select
from bokeh.models.callbacks import CustomJS
from bokeh.io import output_notebook, push_notebook, output_file, save
from bokeh.layouts import gridplot, column, row
import bokeh
from bokeh import plotting, models
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram
from keras import backend as K
from keras.layers.core import Lambda
import tensorflow as tf


def get_order(linkage):
    return dendrogram(linkage, no_plot=True)['leaves']

def is_num(x):
	try:
		int(x)
		return True
	except:
		return False

def cos_sim_garads(model):
    # split
    # TODO:vector 以外に対応させる reshapeで対応
    after_split_shape = [-1, 1]
    # after_split_shape = (reduce(lambda a, b: a*b,model.output.shape.as_list()[1:]), 1) # (-1, 1)
    a = Lambda(lambda x: tf.reshape(x[0], after_split_shape))(model.output)
    b = Lambda(lambda x: tf.reshape(x[1], after_split_shape))(model.output)
    """
    after_split_shape = tuple(model.output.shape[1:].as_list() + [1])
    a = Lambda(lambda x: tf.expand_dims(x[0], 1), output_shape=lambda input_shape: after_split_shape)(model.output)
    b = Lambda(lambda x: tf.expand_dims(x[1,:], 1), output_shape=lambda input_shape: after_split_shape)(model.output)
    """
    # cossim
    a_dot_b = tf.matmul(a, b, transpose_a=True)
    a_abs = tf.sqrt(tf.matmul(a, a, transpose_a=True))
    b_abs = tf.sqrt(tf.matmul(b, b, transpose_a=True))
    cossim = tf.div(a_dot_b, tf.multiply(a_abs, b_abs))

    grad_tensor = K.gradients(cossim, model.input)[0]  
    grad_func = K.function([model.input], [grad_tensor])
    cossim_function = K.function([model.input], [cossim])

    def get_grad_sim(input1_input2):
        # return K.function([ae_mnist.input], [cossim])([input1_input2])[0]
        return grad_func([input1_input2])[0], cossim_function([input1_input2])[0][0][0]
    return get_grad_sim


def trans_data_mnist(data):
    n_sample = data.shape[0]
    """
    img = np.empty((n_sample, n_sample), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((n_sample, n_sample, 4))
    for i in range(N):
        for j in range(N):
            view[i, j, 0] = int(i/N*255)
            view[i, j, 1] = 158
            view[i, j, 2] = int(j/N*255)
            view[i, j, 3] = 255
    """
    # data = np.abs(data)
    return np.flip(data.reshape(n_sample, 28, 28), axis=1)


def item_vis_for_img(datasource, labels=None, d_shape=28*28):
    """
    item_vis テストデータの可視化手法: test_data, sali=>figures layout,list of columndatasource, tap_callback
    callbackについて
    args=dict(ds_clustermap=ds_clustermap, 
        ds_list_item_vis=ds_list_item_vis, ds_test_data=ds_test_data, n_sample=n_sample)
    """
    figures = {}
    fig_titles = ['test_data', 'test_data', 'saliency_pos', 'saliency_pos', 'saliencies_neg', 'saliency_neg']
    l = [datasource.data['test_data'][0], datasource.data['test_data'][1],
         datasource.data['saliency_pos'][0, 1], datasource.data['saliency_pos'][1, 0],
         datasource.data['saliency_neg'][0, 1], datasource.data['saliency_neg'][1, 0]]
    ds_list_item_vis = {}
    for i in range(6):
        ds_list_item_vis[i] = ColumnDataSource(data={'z':[l[i]]})
        figures[i] = figure(title=fig_titles[i], plot_width=200, plot_height=200, x_range=(0, 10), y_range=(0, 10))
        figures[i].image('z', source=ds_list_item_vis[i], x=0, y=0, dw=10, dh=10)
    layout_item_vis = gridplot([[figures[r+2*c] for r in range(2)] for c in range(3)])
    callback_args = dict(d_shape=d_shape)
    tap_callback = """
    var index = ds_clustermap.selected['1d'].indices;
    xlabel_idx = ds_clustermap.data.xidx[index].split('_').map((s)=>Number(s));
    ylabel_idx = ds_clustermap.data.yidx[index].split('_').map((s)=>Number(s));
    ds_list_item_vis[0].data.z[0] = ds_test_data.data.test_data.slice(d_shape*xlabel_idx[1], d_shape*(xlabel_idx[1]+1));
    ds_list_item_vis[0].change.emit();
    ds_list_item_vis[1].data.z[0] = ds_test_data.data.test_data.slice(d_shape*ylabel_idx[1], d_shape*(ylabel_idx[1]+1));
    ds_list_item_vis[1].change.emit();
    ds_list_item_vis[2].data.z[0] = ds_test_data.data.saliency_pos.slice(d_shape*(n_sample*xlabel_idx[1]+ylabel_idx[1]), d_shape*(n_sample*xlabel_idx[1]+ylabel_idx[1]+1));
    ds_list_item_vis[2].change.emit();
    ds_list_item_vis[3].data.z[0] = ds_test_data.data.saliency_pos.slice(d_shape*(n_sample*ylabel_idx[1]+xlabel_idx[1]), d_shape*(n_sample*ylabel_idx[1]+xlabel_idx[1]+1));
    ds_list_item_vis[3].change.emit();
    ds_list_item_vis[4].data.z[0] = ds_test_data.data.saliency_neg.slice(d_shape*(n_sample*xlabel_idx[1]+ylabel_idx[1]), d_shape*(n_sample*xlabel_idx[1]+ylabel_idx[1]+1));
    ds_list_item_vis[4].change.emit();
    ds_list_item_vis[5].data.z[0] = ds_test_data.data.saliency_neg.slice(d_shape*(n_sample*ylabel_idx[1]+xlabel_idx[1]), d_shape*(n_sample*ylabel_idx[1]+xlabel_idx[1]+1));
    ds_list_item_vis[5].change.emit();
    """
    return layout_item_vis, ds_list_item_vis, tap_callback, callback_args


def item_vis_for_multidim(datasource, labels, d_shape=28*28):
    """
    item_vis テストデータの可視化手法: test_data, sali=>figures layout,list of columndatasource, tap_callback
    callbackについて
    args=dict(ds_clustermap=ds_clustermap, 
        ds_list_item_vis=ds_list_item_vis, ds_test_data=ds_test_data, n_sample=n_sample)
    """
    n_dim = len(labels)
    figures = {}
    fig_titles = ['test_data', 'test_data', 'saliency_pos', 'saliency_pos', 'saliencies_neg', 'saliency_neg']
    l = [datasource.data['test_data'][0], datasource.data['test_data'][1],
         datasource.data['saliency_pos'][0, 1], datasource.data['saliency_pos'][1, 0],
         datasource.data['saliency_neg'][0, 1], datasource.data['saliency_neg'][1, 0]]
    ds_list_item_vis = {}
    for i in range(6):
        ds_list_item_vis[i] = ColumnDataSource(data=dict(xidx=labels, yidx=l[i]))
        figures[i] = figure(title=fig_titles[i], plot_width=200, plot_height=200,
               tools="hover", x_range=labels)
        figures[i].vbar(x='xidx', top='yidx', width=0.9, source=ds_list_item_vis[i])
        figures[i].select_one(HoverTool).tooltips = [('label, val', '@xidx, @yidx')]
        figures[i].xaxis.major_label_orientation = -np.pi/3
    layout_item_vis = gridplot([[figures[r+2*c] for r in range(2)] for c in range(3)])
    callback_args = dict(n_dim=n_dim, n_sample=len(datasource.data['test_data']))
    # d[fig_idx][label] = max min
    tap_callback = """
    // tap したindex の取得
    var index = ds_clustermap.selected['1d'].indices;
    xlabel_idx = ds_clustermap.data.xidx[index].split('_').map((s)=>Number(s));
    ylabel_idx = ds_clustermap.data.yidx[index].split('_').map((s)=>Number(s));
    // 配列から必要な値を取得
    const x_pos = [n_dim*xlabel_idx[1], n_dim*(xlabel_idx[1]+1)]
    const y_pos = [n_dim*ylabel_idx[1], n_dim*(ylabel_idx[1]+1)]
    const d_size = n_dim*n_sample;
    console.log(x_pos, y_pos, d_size, ds_test_data.data.test_data)
    const data_array = [
    ds_test_data.data.test_data.slice(x_pos[0], x_pos[1]),
    ds_test_data.data.test_data.slice(y_pos[0], y_pos[1]),
    ds_test_data.data.saliency_pos.slice(d_size*xlabel_idx[1]+y_pos[0], d_size*xlabel_idx[1]+y_pos[1]),
    ds_test_data.data.saliency_pos.slice(d_size*ylabel_idx[1]+x_pos[0], d_size*ylabel_idx[1]+x_pos[1]),
    ds_test_data.data.saliency_neg.slice(d_size*xlabel_idx[1]+y_pos[0], d_size*xlabel_idx[1]+y_pos[1]),
    ds_test_data.data.saliency_neg.slice(d_size*ylabel_idx[1]+x_pos[0], d_size*ylabel_idx[1]+x_pos[1])
    ];
    // dsの書き換え
    for(var i=0;i<6;i++){
        ds_list_item_vis[i].data.yidx = data_array[i];
        ds_list_item_vis[i].change.emit()
    }
    """
    return layout_item_vis, ds_list_item_vis, tap_callback, callback_args
    

def item_vis_for_mat(datasource, labels, d_shape=28*28):
    """
    TODO:どことなく汚いから描き直す
    item_vis テストデータの可視化手法: test_data, sali=>figures layout,list of columndatasource, tap_callback
    callbackについて
    args=dict(ds_clustermap=ds_clustermap, 
        ds_list_item_vis=ds_list_item_vis, ds_test_data=ds_test_data, n_sample=n_sample)
    """
    # 前処理として、cat2colordict, categories, n_datas
    colors = bokeh.palettes.Category20
    colors[1] = ['#1f77b4']
    colors[2] = ['#1f77b4', '#aec7e8']
    is_mat = len(datasource.data['test_data'][0].shape) == 2
    cat2colordict, categories = dict(), []
    for i, label in enumerate(labels):
        vals = datasource.data['test_data'][:,:,i].flatten() if is_mat else datasource.data['test_data'][:,i].flatten()
        if is_num(vals[0]):
            continue
        vals = set(vals)
        categories.append(label)
        if len(vals) > 20:
            cat2colordict[label] = {k: '#ffff00' for k in vals} # 多すぎるので真っ黄色
        cat2colordict[label] = {k:v for k, v in zip(vals, colors[len(vals)])}
    # 正規化用の辞書作成
    figidx2max_min_dict = {}
    for i, d in enumerate([datasource.data['test_data'], datasource.data['saliency_pos'], datasource.data['saliency_neg']]):
        max_min_dict = {}
        for j, label in enumerate(labels):
            if len(d.shape)==4:
                vals = d[:, :, :, j].flatten()
            elif len(d.shape)==3:
                vals = d[:, :, j].flatten()
            elif len(d.shape)==2:
                vals = d[:, j].flatten()
            if not is_num(vals[0]):
                continue
            if vals.max() == vals.min():
                max_min_dict[label] = [1, 0] # 正規化するようなので/0を裂ける
            else:
                max_min_dict[label] = [vals.max(), vals.min()]
        figidx2max_min_dict[2*i] = max_min_dict
        figidx2max_min_dict[2*i+1] = max_min_dict
    n_datas = []
    for i in range(len(datasource.data['test_data'])):
        n_datas.append(len(datasource.data['test_data'][i].flatten())//len(labels))
    figures = {}
    fig_titles = ['test_data', 'test_data', 'saliency_pos', 'saliency_pos', 'saliencies_neg', 'saliency_neg']
    l = [datasource.data['test_data'][0], datasource.data['test_data'][1],
         datasource.data['saliency_pos'][0, 1], datasource.data['saliency_pos'][1, 0],
         datasource.data['saliency_neg'][0, 1], datasource.data['saliency_neg'][1, 0]]
    ds_list_item_vis = {}
    for i in range(6):
        N = len(l[i].flatten()) // len(labels)
        ds_list_item_vis[i] = ColumnDataSource(data=dict(
            xidx=[label for j in range(N) for label in labels], # labels, labels, labels ...
            yidx=[j for j in range(N) for label in labels], # 0,0,0... 1,1,1 ...
            colors=[cat2colordict[label][val]  if label in categories else 'gray' for (label, val) in zip(cycle(labels), l[i].flatten())],
            alphas=[1 if label in categories else (val-figidx2max_min_dict[i][label][1])/(figidx2max_min_dict[i][label][0]-figidx2max_min_dict[i][label][1]) for (label, val) in zip(cycle(labels), l[i].flatten())],
            vals=l[i].flatten()))
        figures[i] = figure(title=fig_titles[i], plot_width=200, plot_height=200,
               x_axis_location="above", tools="hover",
               x_range=labels, y_range=(0, N))
        figures[i].grid.grid_line_color = None
        figures[i].axis.axis_line_color = None
        figures[i].axis.major_tick_line_color = None
        figures[i].axis.major_label_text_font_size = "5pt"
        figures[i].axis.major_label_standoff = 0
        figures[i].xaxis.major_label_orientation = np.pi/3
        figures[i].rect('xidx', 'yidx', 0.9, 0.9, source=ds_list_item_vis[i],
           color='colors', alpha='alphas', line_color=None)
        figures[i].select_one(HoverTool).tooltips = [
            ('label, idx', '@xidx, @yidx'), 
            ('val', '@vals'),
        ]
    layout_item_vis = gridplot([[figures[r+2*c] for r in range(2)] for c in range(3)])
    callback_args = dict(n_datas=n_datas, f_items=figures, cat2colordict=cat2colordict, categories=categories, n_dim=len(labels), labels=labels, figidx2max_min_dict=figidx2max_min_dict)
    # d[fig_idx][label] = max min
    tap_callback = """
    // tap したindex の取得
    var index = ds_clustermap.selected['1d'].indices;
    xlabel_idx = ds_clustermap.data.xidx[index].split('_').map((s)=>Number(s));
    ylabel_idx = ds_clustermap.data.yidx[index].split('_').map((s)=>Number(s));
    // 配列から必要な値を取得
    const reducer = (acc, cur) => cur*n_dim + acc;
    const x_pos = [n_datas.slice(0, xlabel_idx[1]).reduce(reducer, 0), n_datas.slice(0, xlabel_idx[1]+1).reduce(reducer, 0)];
    const y_pos = [n_datas.slice(0, ylabel_idx[1]).reduce(reducer, 0), n_datas.slice(0, ylabel_idx[1]+1).reduce(reducer, 0)];
    const d_size = n_datas.reduce(reducer, 0);
    console.log(x_pos, y_pos, d_size, n_datas, n_dim);
    const range = (n) => Array.from(Array(n), (v, k) => k)
    const data_array = [
    ds_test_data.data.test_data.slice(x_pos[0], x_pos[1]),
    ds_test_data.data.test_data.slice(y_pos[0], y_pos[1]),
    ds_test_data.data.saliency_pos.slice(d_size*xlabel_idx[1]+y_pos[0], d_size*xlabel_idx[1]+y_pos[1]),
    ds_test_data.data.saliency_pos.slice(d_size*ylabel_idx[1]+x_pos[0], d_size*ylabel_idx[1]+x_pos[1]),
    ds_test_data.data.saliency_neg.slice(d_size*xlabel_idx[1]+y_pos[0], d_size*xlabel_idx[1]+y_pos[1]),
    ds_test_data.data.saliency_neg.slice(d_size*ylabel_idx[1]+x_pos[0], d_size*ylabel_idx[1]+x_pos[1])
    ];
    // dsの書き換え
    for(var i=0;i<6;i++){
        const N = data_array[i].length/n_dim
        // yrangeの書き換え
        //f_items[i].y_range.factors=range(N)
        //ds_list_item_vis[i].data.xidx = range(N).reduce((acc, cur)=>acc.concat(labels), []);
        //ds_list_item_vis[i].data.yidx = labels.reduce((acc, cur)=>acc.concat(range(N)), []);
        //ds_list_item_vis[i].data.colors = data_array[i].map((cur, idx)=>{
          //   const label=labels[idx%n_dim]
            // return (categories.includes(label)) ? cat2colordict[label][cur] : 'gray'
        //});
        ds_list_item_vis[i].data.alphas = data_array[i].map((cur, idx)=>{
             const label=labels[idx%n_dim];
             return (categories.includes(label)) ? 1 : (cur-figidx2max_min_dict[i][label][1])/(figidx2max_min_dict[i][label][0]-figidx2max_min_dict[i][label][1]);
        });
        ds_list_item_vis[i].data.vals = data_array[i]
        ds_list_item_vis[i].change.emit()
    }
    """
    return layout_item_vis, ds_list_item_vis, tap_callback, callback_args



def model_vis(model, test_data, test_label, item_vis=item_vis_for_img,
 item_vis_label=None, trans_data=(lambda x: x), trans_data_for_saliency=None,
 get_saliency=cos_sim_garads, output_path=None, output_title='Bokeh plot'):
    """
    args:
        モデル: 特徴量を返すモデル
        テストデータ: (num_sample, *data_shape)
        item_vis テストデータの可視化手法: test_data, sali=>figures layout,list of columndatasource, tap_callback
        可視化前の変換: test_data set=>~
        グラディエントの取得関数: model=>(np.array([input_a, input_b])=>[saliency_a, saliency_b], sim_score)
    
    """
    if output_path is not None:
        output_file(output_path, title=output_title)
    elif 'ipykernel' in sys.modules:
        output_notebook()
    # prepare data
    n_sample, *d_shape = test_data.shape
    get_saliency_f = get_saliency(model)
    test_data_toshow = trans_data(test_data)
    d_toshow_shape = test_data_toshow.shape[1:]
    saliencies = np.zeros((n_sample, n_sample, *d_shape))
    saliencies_toshow = np.zeros((n_sample, n_sample, *d_toshow_shape))
    sim_mat = np.zeros((n_sample, n_sample))
    label2num = {k: v for v, k in enumerate(set(test_label))}
    for i in tqdm(range(n_sample), leave=False):
        for j in range(n_sample):
            if j < i:
                continue
            data1, data2 = (test_data[i], test_data[j])
            (sal1, sal2), score = get_saliency_f(np.array([data1, data2]))
            saliencies[i, j] = sal1
            saliencies[j, i] = sal2
            sim_mat[i, j] = score
            sim_mat[j, i] = score
    if trans_data_for_saliency is None:
        for i in range(n_sample):
            saliencies_toshow[i] = trans_data(saliencies[i])
    else:
        saliencies_toshow = trans_data_for_saliency(saliencies)
    saliencies_pos_toshow = np.clip(saliencies_toshow, 0, np.inf)
    saliencies_neg_toshow = -np.clip(saliencies_toshow, -np.inf, 0) # -?
    cluster_order_idx = get_order(linkage(sim_mat))
    order_str = list(map(lambda idx:f'{test_label[idx]}_{idx}', cluster_order_idx))
    cluster_order = (list(reversed(order_str)), order_str)
    dict_order = (sorted(order_str, reverse=False), sorted(order_str, reverse=True))
    
    # prepare datasource
    ds_test_data = ColumnDataSource(data=dict(test_data=test_data_toshow,
     saliency=saliencies_toshow, saliency_pos=saliencies_pos_toshow, saliency_neg=saliencies_neg_toshow))
    # TODO:配色数値に対して自動化
    colormap = bokeh.palettes.Category10[10]
    xidx, xlabel, yidx, ylabel, color, alpha = [[] for i in range(6)]
    for i in range(n_sample):
        for j in range(n_sample):
            label_i = test_label[i]
            label_j = test_label[j]
            xidx.append(f'{test_label[i]}_{i}')
            yidx.append(f'{test_label[j]}_{j}')
            xlabel.append(label_i)
            ylabel.append(label_j)
            alpha.append(max(0, 2*min(1, sim_mat[i, j])-1)) # float32でcossimが>1のものがある
            if label_i == label_j:
                color.append(colormap[label2num[label_i]])
            else:
                color.append('lightgray')
    color_bw = ['gray' for i in range(n_sample*n_sample)]
    ds_clustermap = ColumnDataSource(data=dict(
        xidx=xidx,
        yidx=yidx,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=color,
        alphas=alpha,
        sims=sim_mat.flatten()))

    # saliency map
    layout_item_vis, ds_list_item_vis, tap_callback, tap_callback_args = item_vis(ds_test_data, item_vis_label)
    
    # clustermap
    f_cluster = figure(title="model vis",
               x_axis_location="above", tools="hover,save,tap",
               x_range=cluster_order[0], y_range=cluster_order[1])
    f_cluster.plot_width = 800
    f_cluster.plot_height = 800
    f_cluster.grid.grid_line_color = None
    f_cluster.axis.axis_line_color = None
    f_cluster.axis.major_tick_line_color = None
    f_cluster.axis.major_label_text_font_size = "5pt"
    f_cluster.axis.major_label_standoff = 0
    f_cluster.xaxis.major_label_orientation = np.pi/3

    renderer = f_cluster.rect('xidx', 'yidx', 0.9, 0.9, source=ds_clustermap,
           color='colors', alpha='alphas', line_color=None, selection_fill_alpha=1, selection_line_color='red')
    renderer.nonselection_glyph = None

    f_cluster.select_one(HoverTool).tooltips = [
        ('labels', '@xlabel, @ylabel'),
        ('similarity', '@sims'),
    ]
    tap_callback_args.update(dict(ds_clustermap=ds_clustermap, ds_list_item_vis=ds_list_item_vis, ds_test_data=ds_test_data, n_sample=n_sample))
    f_cluster.select_one(TapTool).callback = CustomJS(args=tap_callback_args, code=tap_callback)
    # button
    toggle = Toggle(label="Colorful", button_type="success")
    toggle.callback = CustomJS(args=dict(ds_clustermap=ds_clustermap, color=color, color_bw=color_bw), code="""
    if (cb_obj.label=='Colorful'){
        cb_obj.label = 'Monochrome';
        ds_clustermap.data.colors = color_bw;
    } else {
        cb_obj.label = 'Colorful';
        ds_clustermap.data.colors = color;
    } 
    ds_clustermap.change.emit()
    """)
    select = Select(title="Order:", value="hierarchical_clustering", options=["hierarchical_clustering", "dict_order"])
    select.callback = CustomJS(args=dict(f_cluster=f_cluster, dict_order=dict_order, cluster_order=cluster_order), code="""
    if (cb_obj.value=='hierarchical_clustering'){
        var order = cluster_order
    } else if(cb_obj.value=='dict_order') {
        var order = dict_order
    } 
    f_cluster.x_range.factors=order[0]
    f_cluster.y_range.factors=order[1]
    """)
    layout_buttons = gridplot([[toggle], [select]])
    layout_item_buttons = column(layout_item_vis, layout_buttons)
    layout_doc = row(f_cluster, layout_item_buttons)
    show(layout_doc) # show the plot
