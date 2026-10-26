# Digital Twin–Based Wireless Positioning

Scene generation, radio simulation, and wireless positioning. This public repository is updated continuously as our research progresses.

[Setup](docs/setup.md) · [Quickstart](docs/quickstart.md) · [Scene sources](docs/scene-sources.md)

**Scenes:** Geo2SigMap, Overture height enrichment, LiDAR/DEM, BostonTwin, and Mitsuba XML.

![Two generated New York scenes](docs/assets/nyc-scene-comparison.png)

*New York: OSM/Overture (left), LiDAR (right). © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). [Source credits](docs/visual-examples.md).*

**Pipeline:** scene → ray tracing → channel responses → beam-delay fingerprints → positioning.

![Simulated beam-delay fingerprints](docs/assets/synthetic-fingerprints.png)

*Fingerprints from the runnable example. The current release includes weighted k-NN positioning.*

![Positioning error maps for two CNN models](docs/assets/positioning-results.png)

*Two CNN models, 3,841 test locations, three seeds. [Study details](docs/results.md). CNN code and weights are forthcoming.*

**Contributors:** [Canan Cebeci](https://github.com/canancebeci) and [Josef Ullrich](https://github.com/josefullrich).

**License:** [CC BY-NC 4.0](LICENSE). [Third-party terms](NOTICE.md).
