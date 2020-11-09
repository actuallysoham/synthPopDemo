
var polygonID = "";

mapboxgl.accessToken = 'pk.eyJ1IjoiYWRpdGlqYWluODE2IiwiYSI6ImNrZ2pwaThvbzB4cWgzM3FuNmh2ZGp0czkifQ.E_LcBn8FkcViW47rpGozTw';
    var map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/light-v10',
        center: [79, 20],
        zoom: 3
    });

map.on('load', function () {
    // Add a source for the state polygons.
    map.addSource('states', {
                    'type': 'geojson',
                    'data':DATA,  //DATA.js has the stanford geoJSON
                    
                });
 
    // Add a layer showing the state polygons.
    map.addLayer({
        'id': 'states-layer',
        'generateId': true,
        'type': 'fill',
        'source': 'states',
        'paint': {
        'fill-color': 'rgba(200, 100, 240, 0.4)',
        'fill-outline-color': 'rgba(200, 100, 240, 1)'
            }
    });

 
    // When a click event occurs on a feature in the states layer, open a popup at the
    // location of the click, with description HTML from its properties.
    map.on('click', 'states-layer', function (e) {
        polygonID = e.features[0].properties.laa;
        console.log(polygonID);

    new mapboxgl.Popup()
        .setLngLat(e.lngLat)
        .setHTML("<b>"+ e.features[0].properties.nam+ "</b><br>" +e.features[0].properties.laa)
        .addTo(map);

    });

    map.on('mousehover', 'states-layer', function (e) {
        var popup = new mapboxgl.Popup({
        closeOnClick: false,
        offset: [0, -15]
        })
    popup.setLngLat(e.lngLat)
        .setHTML("<b>"+ e.features[0].properties.nam+ "</b><br>" +e.features[0].properties.laa)
        .addTo(map);
   
    });
 
    // Change the cursor to a pointer when the mouse is over the states layer.
    map.on('mouseenter', 'states-layer', function () {
    map.getCanvas().style.cursor = 'pointer';

    
    });
 
    // Change it back to a pointer when it leaves.
    map.on('mouseleave', 'states-layer', function () {
    map.getCanvas().style.cursor = '';
        });
});

document.getElementById("submit").onclick = () => {getData()};
document.getElementById("submitage").onclick = () => {getDataAge()};
document.getElementById("submitsub").onclick = () => {getDataSub()};

function getData(){
    
        // .post('http://127.0.0.1:1234/poly',e.features[0].properties)
        axios.get('/poly', {
            params : {
                name: polygonID
            },
            responseType: 'blob',
        }).then((response) => {
          const url = window.URL.createObjectURL(new Blob([response.data]));
          const link = document.createElement('a');
          link.href = url;
          link.setAttribute('download', 'subpop.csv');
          document.body.appendChild(link);
          link.click();})

}

function getDataAge(){
    
        // .post('http://127.0.0.1:1234/poly',e.features[0].properties)
        axios.get('/polyage', {
            params : {
                name: polygonID
            },
            responseType: 'blob',
        }).then((response) => {
          const url = window.URL.createObjectURL(new Blob([response.data]));
          const link = document.createElement('a');
          link.href = url;
          link.setAttribute('download', 'subpop.csv');
          document.body.appendChild(link);
          link.click();})

}

function getDataSub(){
    
        // .post('http://127.0.0.1:1234/poly',e.features[0].properties)
        axios.get('/polysub', {
            params : {
                name: polygonID
            },
            responseType: 'blob',
        }).then((response) => {
          const url = window.URL.createObjectURL(new Blob([response.data]));
          const link = document.createElement('a');
          link.href = url;
          link.setAttribute('download', 'subpop.csv');
          document.body.appendChild(link);
          link.click();})

}