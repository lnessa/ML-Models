# Pebble Hotspot

## Setup (On MacOS, may differ on Windows/Linux)

- Clone this repository and navigate to toor directory
- Setup virtual environment `$ python3 -m venv venv`
- Activate virtual environmnet `$ source venv/bin/activate`
- Install requirements (may take a while) `$ pip3 install torch pandas matplotlib`

## Useful commands

- Train UNET `$ python3 -m Model train_unet`
- Train Encoder `$ python3 -m Model train_encoder`
- Demo UNET `$ python3 -m Model unet_demo`
- Demo Encoder `$ python3 -m Model encoder_demo`
