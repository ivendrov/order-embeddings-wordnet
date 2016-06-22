

// globals
var synset_names;
var hypernyms;
var db = {};
var datasets_index = [];
var current_methods_dict = [];
var methods = []; // keeps track of currently displayed methods
var cur_method = [];
var dataset_select = [];
var image_urls;
var image2cap = {};
var caption_texts;


function init() {
    console.log("Loading page");

    d3.json("static/index.json", function (data) {
        datasets_index = data;
        var datasets = [];
        for (var dataset in datasets_index){
            if (datasets_index.hasOwnProperty(dataset)) {
                datasets.push(dataset);
            }
        }

        dataset_select = d3.select("#data_select").select("[name=dataset]");

        dataset_select.selectAll("option")
            .data(datasets)
            .enter()
            .append("option")
            .attr("name", function(d){
                return d;
            })
            .text(function(d){
                return d;
            });
        dataset_select.on("change",update_methods);
        update_methods();

        // get URL GET params
        var queryDict = {};
        queryDict["method"] = [];
        location.search.substr(1).split("&").forEach(function(item) {
            var split = item.split("=");
            var name = split[0];
            var value = split[1];
            if (name == "method")
                queryDict["method"].push(value);
            else
                queryDict[name] = value
        });
        if (queryDict.hasOwnProperty("dataset") && queryDict.hasOwnProperty("method")){
            console.log("Loading page with form");
            dataset_select.node().value = decodeURIComponent(queryDict["dataset"]);
            update_methods();

            var method_select = d3.select("#data_select").select("[name=method]").node();

            for (var i =0; i < method_select.options.length; i++) {
                var option = method_select.options[i];
                if (queryDict["method"].indexOf(option.value) >= 0) {
                    option.selected = true;
                }
            }
            load();
        }

    });
}

function update_methods(){
    methods_list = datasets_index[dataset_select.node().value];
    methods_list.sort();

    var method_select = d3.select("#data_select").select("[name=method]");
    method_select.selectAll("*").remove();
    method_select
        .selectAll("option")
        .data(methods_list)
        .enter()
        .append("option")
        .attr("name", function(d){
        return d;
        })
        .text(function(d){
            return d;
        });
}
function selectedMethods() {
    var form = document.getElementById("data_select");
    var methods = [];
    // add all selected methods to DB
    for (var i =0; i < form.method.options.length; i++){
        var option = form.method.options[i];
        if (option.selected){
            methods.push(option.value);
        }
    }
    return methods;
}
function selectedDataset() {
    var form = document.getElementById("data_select");
    return form.dataset.value;
}

function load() {
  // populate datasets
    var dataset = selectedDataset();
    var methods = selectedMethods();


    current_methods_dict = datasets_index[dataset];

    // load images
    $.ajax({
        dataType: "json",
        url: 'static/' + dataset + '/hypernyms.json',
        async: false,
        success: function(data) {
            hypernyms = data;
        }
    });

    // TODO TODO CHANGE TO hypernyms.json and synset_names.json
    // load captions
    $.ajax({
        dataType: "json",
        url: 'static/' + dataset + '/synset_names.json',
        async: false,
        success: function(data) {
            synset_names = data;
        }
    });

    datasets_loaded = 0;
    for (var i = 0; i < methods.length; i++){
        var method = methods[i];
        var path = 'static/' + dataset + '/' + method;
        console.log("Loading dataset from " + path);
        loadDataset(path, method, methods.length);
    }


    document.getElementById("dataset_load_error").innerHTML = "";
    displayResults()

}

/** display the table of statistics */
function displayStats(colors){
    console.log("Generating stats");
    var statsTable = d3.select("#stats_table");
    statsTable.selectAll("*").remove();

    // generate data
    var methods = Object.keys(db);
    var stat_names = ["Method"];
    var data = [];
    var best_value = [];

    for (var i = 0; i < db[methods[0]].stats.length; i++){
        var stat = db[methods[0]].stats[i];
        stat_names.push(stat.name);
        if (stat.name != "median_rank")
            best_value.push(0);
        else
            best_value.push(1000000);
    }
    for (method in db){

        var stats = db[method].stats;
        var method_data = [method];
        method_data = method_data.concat(stats);
        data.push(method_data);

        // compute best values for all statistics
        for (var i = 0; i < stats.length; i++){
            stat = stats[i];
            if (stat.name != "median_rank") {
                if (stat.value > best_value[i]) {
                    best_value[i] = stat.value;

                }
            } else {
                if (stat.value < best_value[i]) {
                    best_value[i] = stat.value;
                }
            }
        }
    }


    // table header
    statsTable.append("thead")
        .append("tr")
        .selectAll("th")
        .data(stat_names)
        .enter()
        .append("th")
        .text(function (d) {
            return d;
        });


    // table body
    var tbody = statsTable.append("tbody");
    var rows = tbody.selectAll("tr")
                    .data(data)
                    .enter()
                    .append("tr");

    rows.selectAll("td")
        .data(function (d) {
            return d;
        })
        .enter()
        .append("td")
        .style("color", function(d, i){
            if (i == 0){
                var color = colors[methods.indexOf(d)];
                return color;
            } else {
                return null;
            }
        })
        .style("text-align", "center")
        .html(function(d, i){
            var text = [];
            if (i == 0){
               return "<b>" + d + "</b>";
            } else if (i == 1) {
                text = d3.format(" >4g")(d.value);
            } else {
                text = d3.format(".3f")(d.value);
            }
            if (best_value[i-1] == d.value){
                text = "<u>" + text + "</u>";
            }
            return text;

        });


}

function displayResults() {
    // display caption search form
    d3.select("#caption_search").style("display", "block");
    // generate random colors for each method
    var colors = [];
    methods = [];
    for (method in db){
        colors.push(d3.rgb('#'+Math.random().toString(16).substr(-6)).darker(1));
        methods.push(method);
    }

    displayStats(colors);

    // display hyperparams
    d3.select("#hyperparams").selectAll("div").remove();
    d3.select("#hyperparams").selectAll("div")
        .data(methods)
        .enter()
        .append("div")
        .style("color", function(d, i){
            return colors[i];
        })
        .append("pre")
        .text(function(d){
            return JSON.stringify(db[method].hyperparams, null, 2);
        });

    var method = methods[0];
    var embeddings = db[method].embeddings;




    var nDims = embeddings[0].length;


     // TODO let the user select these
    var dim1 = 0;
    var dim2 = 1;
    var nWords = 1000;

    var trans = d3.transpose(embeddings.slice(0, nWords));
    synset_names = synset_names.slice(0, nWords);

    var X = trans[dim1];
    var Y = trans[dim2];

    var words_main = d3.select("#words");

    words_main.selectAll("div").remove();

    // add svgs to new charts

    var margin = {top: 20, right: 30, bottom: 50, left: 60}; // Mike's margin convention

    var w = 1500 - margin.left - margin.right,
        h = 1500 - margin.top - margin.bottom;

    var chart = words_main
        .append("svg")
        .attr("width", w + margin.left + margin.right)
        .attr("height", h + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("width", w)
        .attr("height", h)
        .attr("class", "chart");

    var xScale = d3.scale.linear()
        .domain([d3.min(X), d3.max(X)])
        .range([0, w]);
    var yScale = d3.scale.linear()
        .domain([d3.min(Y), d3.max(Y)])
        .range([h, 0]);

    // draw lines
    var selected_hypernyms = [];
    for (var i = 0; i < hypernyms.length; i++){
        var pair = hypernyms[i];
        if (pair[0] <= nWords && pair[1] <= nWords){
            selected_hypernyms.push([pair[0]-1, pair[1]-1]); // convert to 0-based
        }
    }


    var arrows = chart
        .selectAll(".arrow")
        .data(selected_hypernyms)
        .enter()
        .append("line")
        .attr("x1", function(d){ return xScale(X[d[1]]);})
        .attr("x2", function(d){ return xScale(X[d[0]]);})
        .attr("y1", function(d){ return yScale(Y[d[1]]);})
        .attr("y2", function(d){ return yScale(Y[d[0]]);})
        .style("opacity", 0.3)
        .style("stroke", function(d){
            if (X[d[1]] > X[d[0]] || Y[d[1]] > Y[d[0]]) {
                return d3.rgb(255,0,0); // violation, draw in red
            } else {
                return d3.rgb(0,0,0);
            }
        })
        .style("stroke-width", 2);

    arrows
        .append("svg:title")
        .text(function(d){
            return synset_names[d[1]] + " -> "  + synset_names[d[0]];
        });


    // draw words

    var words = chart.selectAll(".word")
        .data(synset_names)
        .enter()
        .append("g")
        .attr("id", function(d, i){ return "word" + i})
        .attr("transform", function(d, i) {
            return "translate(" + xScale(X[i]) + "," + yScale(Y[i]) + ")";
        });

    words
        .append("circle")
            .attr("r", 2);

    words.append("text")
        .text(function(d){ return d})
        .style('font-size', '.5em');


    // draw axes
    chart
        .append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0," + h + ")")
        .each(function() {
            d3.svg.axis()
                .scale(xScale)
                .orient("bottom")
                .ticks(10)
            (d3.select(this));
        });


    chart
        .append("g")
        .attr("class", "axis")
        .each(function(){
             d3.svg.axis()
                .scale(yScale)
                .orient("left")
                .ticks(10)
             (d3.select(this));
        });




}


// Data Loading
function loadDataset(jsonpath, name, wait_until) {
  // ugly hack to prevent caching below ;(
    var jsonmod = jsonpath + '/index.json' + '?sigh=' + Math.floor(Math.random() * 100000);
    console.log(jsonmod);
    $.ajax({
      dataType: "json",
      url: jsonmod,
        async: false,
      success: function (data) {
          if (data == null) {
              document.getElementById("dataset_load_error").innerHTML = "Error: data not found for " + name;
          } else {
              db[name] = data; // assign to global
          }
      }

    });
}
