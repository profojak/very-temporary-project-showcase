// Visualization of Training of Neural Network - Jakub Pícha, Jakub Profota

// Colors
const background_color = '#1e1e1e';
const container_color = '#2d2d2d';
const label_colors = [ '#e5e5e5', '#96472a', '#d8402f', '#992e51', '#755292',
    '#1670b1', '#009da9', '#00894f', '#9abe26', '#efb715' ];

// Neural network data
var nn_data;

// PCA preprocessing
let pca_mean = [], pca_std = [], pca_eigvecs = [];

// Traced point
let trace_id = null;

// Cached embedding
let cached_embedding = {
    'STAR': {},
    'PCA': {},
    'UMAP': {},
    't-SNE': {}
};

// Preprocess PCA embedding
function preprocessPCA(data) {
    pca_mean = Array(data[0].length).fill(0);
    data.forEach(pt => pt.forEach((v,i) => { pca_mean[i] += v; }));
    pca_mean = pca_mean.map(v => v / data.length);
    const centered = data.map(pt => pt.map((v,i) => v - pca_mean[i]));
    pca_std = Array(data[0].length).fill(0);
    centered.forEach(pt => pt.forEach((v,i) => { pca_std[i] += v * v; }));
    pca_std = pca_std.map(v => Math.sqrt(v / data.length));
    const normalized = centered.map(pt => pt.map((v,i) => v / pca_std[i]));
    let cov = numeric.dot(numeric.transpose(normalized), normalized);
    for (let i = 0; i < data[0].length; i++) {
        for (let j = 0; j < data[0].length; j++) {
            cov[i][j] /= (data.length - 1);
        }
    }
    const eigen = numeric.eig(cov);
    const eigvals = eigen.lambda.x;
    const eigvecs = eigen.E.x;

    const sorted = eigvals.map((v,i) => [v,i]).sort((a,b) => b[0] - a[0]).map(pair => pair[1]);
    pca_eigvecs = eigvecs.map(row => sorted.map(j => row[j]));
}

// Embed data using selected reduction method
function embedData(data, dimension = 2) {
    // Obtain selected embedding method
    var selected_embedding = document.querySelector('input[name="embedding"]:checked').value;

    // Star coordinates
    if (selected_embedding === 'STAR') {
        const angles = Array.from({ length: data[0].length },
            (_, i) => i * 2 * Math.PI / data[0].length);
        const axes = angles.map(a => [Math.cos(a), Math.sin(a)]);
        const embedding = Array(data.length).fill().map(() => Array(data[0].length).fill(0));
        for (let i = 0; i < data.length; i++) {
            const point = data[i];
            const projection = [0, 0];
            for (let j = 0; j < point.length; j++) {
                projection[0] += point[j] * axes[j][0] * 15;
                projection[1] += point[j] * axes[j][1] * 15;
            }
            embedding[i] = projection;
        }
        return embedding;
    }

    // PCA
    else if (selected_embedding === 'PCA') {
        const norm = data.map(pt => pt.map((v,i) => (v - pca_mean[i]) / pca_std[i]));
        const embedding = norm.map(pt => {
            let x = 0, y = 0;
            for (let i = 0; i < data[0].length; i++) {
                x += pt[i] * pca_eigvecs[i][0];
                y += pt[i] * pca_eigvecs[i][1];
            }
            return [x * 6, y * 6];
        });
        return embedding;
    }

    // UMAP
    else if (selected_embedding === 'UMAP') {
        const umap_2D = new window.UMAP.UMAP({
            nComponents: dimension,
            nEpochs: 200
        });
        const embedding = umap_2D.fit(data);
        return embedding;
    }

    // t-SNE
    else if (selected_embedding === 't-SNE') {
        const data_subset = Array.from({ length: data[0].length / 25 });
        for (let i = 0; i < data.length; i++) {
            if (i * 25 < data.length) {
                data_subset[i] = data[i * 25];
            }
        }
        let model = new TSNE({ dim: dimension, perplexity: 30, earlyExaggeration: 4,
            learningRate: 100, nIter: 200, metric: 'euclidean' });
        model.init({ data: data_subset, type: 'dense' });
        let [error, iter] = model.run();
        const embedding = model.getOutput();
        return embedding;
    }
}

// Highlight number class on mouseover
function highlightClass(event) {
    // Highlight visualization
    const class_name = event.target.getAttribute('name');
    d3.selectAll('circle')
        .filter(function() { return d3.select(this).classed('trace') !== true; })
        .attr('fill', '#000000');
    d3.selectAll('circle')
        .filter(function() { return d3.select(this).attr('name') === class_name; })
        .attr('fill', label_colors[parseInt(class_name)]);

    // Highlight class label
    d3.select('#class_colors_container').selectAll('div')
        .style('background-color', '#000000')
        .style('color', '#FFFFFF');
    d3.select('#class_colors_container').selectAll('div')
        .filter(function() { return d3.select(this).attr('name') === class_name; })
        .style('background-color', label_colors[parseInt(class_name)])
        .style('color', '#000000');
}

function resetClassLabels(label_colors) {
    d3.select('#class_colors_container').selectAll('div')
        .style('background-color', (d, i) => label_colors[i])
        .style('color', '#000000');
}

// Unhighlight number class on mouseout
function unhighlightClass(embedding, epoch, data, labels) {
    d3.selectAll('circle')
        .filter(function() { return d3.select(this).classed('trace') !== true; })
        .attr('fill', (d, i) => {
            if (embedding.length !== data.length) {
                return label_colors[parseInt(labels[i])];
            }
            return label_colors[parseInt(nn_data['model'][epoch]['real'][i])];
        });
    resetClassLabels(label_colors);
}

// Trace point on click
function tracePoint(event, d, embedding_method, epoch) {
    let class_name;
    let point_number;
    if (event === null) {
        class_name = trace_id[0];
        point_number = trace_id[1];
    } else {
        class_name = parseInt(event.target.getAttribute('name'));
        point_number = event.target.getAttribute('number');
        trace_id = [class_name, point_number];
    }

    let points = [];
    const embeddings = cached_embedding[embedding_method];
    for (const epoch of Object.keys(embeddings)) {
        if (embeddings[epoch] !== null) {
            points.push([embeddings[epoch][point_number], epoch]);
        }
    }

    const svg = d3.select('#visualization');
    const w = +svg.attr('width');
    const h = +svg.attr('height');

    svg.selectAll('.trace-bg').remove();
    svg.selectAll('.trace-bg').data(points).enter().append('circle')
        .classed('trace-bg', true)
        .attr('cx', d => d[0][0] * 25 + w / 2)
        .attr('cy', d => d[0][1] * 25 + h / 2)
        .attr('r', (d, i) => {
            if (d[1] == epoch) {
                return 7;
            }
            return 3;
        })
        .attr('fill', '#000000')
        .attr('opacity', 1);

    svg.selectAll('path').remove();
    for (let i = 0; i < points.length - 1; i++) {
        const [xA, yA] = points[i][0];
        const [xB, yB] = points[i + 1][0];
        const line = d3.line().x(d => d[0] * 25 + w / 2).y(d => d[1] * 25 + h / 2);
        svg.append('path')
            .attr('d', line([[xA, yA], [xB, yB]]))
            .attr('stroke', label_colors[class_name])
            .attr('stroke-width', 1)
            .attr('fill', 'none')
            .attr('opacity', 1);
    }

    svg.selectAll('.trace').remove();
    svg.selectAll('.trace').data(points).enter().append('circle')
        .classed('trace', true)
        .attr('cx', d => d[0][0] * 25 + w / 2)
        .attr('cy', d => d[0][1] * 25 + h / 2)
        .attr('r', (d, i) => {
            if (d[1] == epoch) {
                return 6;
            }
            return 2;
        })
        .attr('fill', label_colors[class_name])
        .attr('opacity', 1);
}

// Draw weights star plot
function drawWeights(epoch) {
    const svg = d3.select('#visualization');
    const w = +svg.attr('width');
    const h = +svg.attr('height');

    const angles = Array.from({ length: 10 },
        (_, i) => i * 2 * Math.PI / 10);
    const axes = angles.map(a => [Math.cos(a), Math.sin(a)]);

    for (let i = 0; i < 10; i++) {
        svg.append('text')
            .attr('x', axes[i][0] * 75 + w - 85)
            .attr('y', axes[i][1] * 75 + h - 105)
            .text(i)
            .style('font-size', '12px')
            .style('fill', '#FFFFFF')
            .style('text-anchor', 'middle')
            .style('dominant-baseline', 'middle');

        svg.append('line')
            .attr('x1', axes[i][0] * 68 + w - 85)
            .attr('y1', axes[i][1] * 68 + h - 105)
            .attr('x2', axes[i][0] * 2 + w - 85)
            .attr('y2', axes[i][1] * 2 + h - 105)
            .attr('stroke', background_color)
            .attr('stroke-width', 1);
        svg.append('line')
            .attr('x1', axes[i][0] * 68 + w - 85)
            .attr('y1', axes[i][1] * 68 + h - 105)
            .attr('x2', axes[(i + 1) % 10][0] * 68 + w - 85)
            .attr('y2', axes[(i + 1) % 10][1] * 68 + h - 105)
            .attr('stroke', background_color)
            .attr('stroke-width', 1);
    }

    svg.append('text')
        .attr('x', w - 85)
        .attr('y', h - 13)
        .text('NN accuracy: ' + parseFloat(nn_data['model'][epoch]['accuracy'] * 100).toFixed(2) + '%')
        .style('font-size', '18px')
        .style('fill', '#FFFFFF')
        .style('text-anchor', 'middle')
        .style('dominant-baseline', 'middle');

    if (trace_id === null) {
        return;
    }

    for (let i = 0; i < 10; i++) {
        svg.append('line')
            .attr('x1', axes[i][0] * nn_data['model'][epoch]['weights'][i][parseInt(trace_id[1])] * 68 + w - 85)
            .attr('y1', axes[i][1] * nn_data['model'][epoch]['weights'][i][parseInt(trace_id[1])] * 68 + h - 105)
            .attr('x2', axes[(i + 1) % 10][0] * nn_data['model'][epoch]['weights'][(i + 1) % 10][parseInt(trace_id[1])] * 68 + w - 85)
            .attr('y2', axes[(i + 1) % 10][1] * nn_data['model'][epoch]['weights'][(i + 1) % 10][parseInt(trace_id[1])] * 68 + h - 105)
            .attr('stroke', label_colors[parseInt(nn_data['model'][epoch]['real'][parseInt(trace_id[1])])])
            .attr('stroke-width', 2);
    }
}

// Update visualization on slider change
function updateVisualization() {
    // Obtain data for current epoch
    const epoch = parseInt(document.getElementById('epoch_slider').value);
    const embedding_method = document.querySelector('input[name="embedding"]:checked').value;
    const data = Array.from({ length: nn_data['model'][epoch]['weights'][0].length },
        (_, i) => Object.values(nn_data['model'][epoch]['weights']).map(arr => arr[i]));

    // Print epoch number
    d3.select('#epoch_label_container').text(epoch + 1)
        .style('color', '#FFFFFF')
        .style('font-size', '18px')
        .style('font-weight', 'bold')
        .style('text-align', 'center')
        .style('line-height', '32px');

    // Embedding reduction
    if (pca_eigvecs.length === 0 && embedding_method === 'PCA') {
        preprocessPCA(Array.from({ length: nn_data['model'][10]['weights'][0].length },
            (_, i) => Object.values(nn_data['model'][10]['weights']).map(arr => arr[i])));
    }
    const embedding = cached_embedding[embedding_method][epoch] || embedData(data);
    let labels = Array.from({ length: nn_data['model'][epoch]['real'].length / 25 });
    if (embedding.length !== data.length) {
        for (let i = 0; i < data.length; i++) {
            if (i * 25 < data.length) {
                labels[i] = nn_data['model'][epoch]['real'][i * 25];
            }
        }
    }

    if (cached_embedding[embedding_method][epoch] === null) {
        cached_embedding[embedding_method][epoch] = embedding;
    }

    // Delete previous visualization
    var visualization = d3.select('#visualization');
    visualization.selectAll('*').remove();

    // Create new visualization
    visualization.selectAll('circle').data(embedding).enter().append('circle')
        .attr('cx', d => d[0] * 25 + d3.select('#visualization').attr('width') / 2)
        .attr('cy', d => d[1] * 25 + d3.select('#visualization').attr('height') / 2)
        .attr('r', 1.5)
        .attr('fill', (d, i) => {
            if (embedding.length !== data.length) {
                return label_colors[parseInt(labels[i])];
            }
            return label_colors[parseInt(nn_data['model'][epoch]['real'][i])];
        })
        .attr('name', (d, i) => {
            if (embedding.length !== data.length) {
                return labels[i];
            }
            return nn_data['model'][epoch]['real'][i];
        })
        .attr('number', (d, i) => {
            if (embedding.length !== data.length) {
                return i * 25;
            }
            return i;
        })
        .attr('opacity', 0.5)
        .on('mouseover', highlightClass)
        .on('mouseout', function(event, d) {
            unhighlightClass(embedding, epoch, data, labels);
        })
        .on('click', function(event, d) {
            visualization.selectAll('path').remove();
            visualization.selectAll('line').remove();
            visualization.selectAll('text').remove();
            tracePoint(event, d, embedding_method, epoch);
            drawWeights(epoch);
        });

    if (trace_id !== null) {
        tracePoint(null, null, embedding_method, epoch);
    }

    drawWeights(epoch);
}

// Load data
async function loadData() {
    // Parse MessagePack
    try {
        const msgpack_resource = await fetch('metadata.msgpack');
        const msgpack_buffer = await msgpack_resource.arrayBuffer();
        nn_data = msgpack.decode(new Uint8Array(msgpack_buffer));
        Object.freeze(nn_data);
    } catch (error) {
        console.error("Error loading data:", error);
    }

    // Setup cached embedding
    for (const embedding of Object.keys(cached_embedding)) {
        for (const epoch of Object.keys(nn_data['model'])) {
            cached_embedding[embedding][epoch] = null;
        }
    }
}

// Initiliaze page
async function initPage() {
    // Document body
    var body = d3.select('body')
        .style('margin', 0)
        .style('padding', 0)
        .style('background-color', background_color)
        .style('display', 'flex')
        .style('justify-content', 'center')
        .style('align-items', 'center')
        .style('min-height', '100vh');

    await loadData();

    // Container for visualization application
    var app_container = body.append('div')
        .attr('id', 'app_container');

    // Actual visualization
    var visualization = app_container.append('svg')
        .attr('id', 'visualization')
        .attr('width', 1024)
        .attr('height', 768)
        .attr('viewBox', [0, 0, 1024, 768])
        .style('background-color', container_color)
        .on('click', function(event) {
            if (event.target.tagName !== 'circle') {
                trace_id = null;
                updateVisualization();
            }
        });

    // Epoch container
    var epoch_container = app_container.append('div')
        .attr('id', 'epoch_container')
        .style('display', 'flex')
        .style('padding', '0px 0px 4px 0px');

    // Epoch slider container
    var epoch_slider_container = epoch_container.append('div')
        .attr('id', 'epoch_slider_container')
        .style('background-color', container_color)
        .style('display', 'flex')
        .style('flex-grow', '1')
        .style('align-items', 'center');

    // Epoch slider
    var epoch_slider = epoch_slider_container.append('input')
        .attr('id', 'epoch_slider')
        .attr('type', 'range')
        .attr('min', 0)
        .attr('max', Object.keys(nn_data['model']).length - 1)
        .attr('value', 0)
        .attr('step', 1)
        .style('width', '100%')
        .style('margin', '0 4px')
        .on('input', updateVisualization);

    // Epoch label container
    var epoch_label_container = epoch_container.append('div')
        .attr('id', 'epoch_label_container')
        .style('width', '32px')
        .style('height', '32px')
        .style('background-color', container_color)
        .style('margin-left', '4px');

    // Embedding container
    var embedding_container = app_container.append('div')
        .attr('id', 'embedding_container')
        .style('background-color', container_color)
        .style('display', 'flex')
        .style('flex-grow', '1')
        .style('align-items', 'center')
        .style('justify-content', 'center');

    // Embedding radio buttons container
    var embedding_radio_container = embedding_container.append('div')
        .attr('id', 'embedding_radio_container')
        .style('height', '32px')
        .style('display', 'flex')
        .style('flex-grow', '1')
        .style('align-items', 'center')
        .style('justify-content', 'left');

    // Embedding radio buttons
    var embedding_radio = embedding_radio_container.selectAll('label')
        .attr('id', 'embedding_radio')
        .data(['STAR', 'PCA', 'UMAP', 't-SNE'])
        .enter()
        .append('label')
        .style('margin', '0 12px')
        .style('color', '#FFFFFF')
        .style('font-size', '18px');

    embedding_radio.append('input')
        .attr('type', 'radio')
        .attr('name', 'embedding')
        .attr('value', d => d)
        .style('margin-right', '6px')
        .on('change', updateVisualization);

    embedding_radio.append('span')
        .text(d => d);

    // Class colors container
    var class_colors_container = embedding_container.append('div')
        .attr('id', 'class_colors_container')
        .style('height', '32px');

    // Class colors labels
    var class_colors_labels = class_colors_container.selectAll('div')
        .data(label_colors)
        .enter()
        .append('div')
        .style('width', '32px')
        .style('height', '32px')
        .style('background-color', (d, i) => d)
        .style('display', 'inline-block')
        .attr('name', (d, i) => i)
        .on('mouseover', highlightClass)
        .on('mouseout', function() {
            updateVisualization();
            resetClassLabels(label_colors);
        })
        .text((d, i) => { if (i < 10) { return i; }
            return String.fromCharCode(65 + i - 10);
        })
        .style('font-size', '18px')
        .style('font-weight', 'bold')
        .style('text-align', 'center')
        .style('line-height', '32px');

    embedding_radio.select('input').attr('checked', d => d === 'STAR' ? true : null);
    updateVisualization();
}

// Wait for all resources to load
window.addEventListener('load', initPage);
