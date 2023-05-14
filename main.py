from flask import Flask, render_template,request, send_file, redirect,url_for,flash
from flask_cors import CORS, cross_origin
from shipment_cost_prediction.pipeline.training_pipeline import Pipeline
from shipment_cost_prediction.pipeline.batch_prediction import Prediction
from shipment_cost_prediction.constant import *
from shipment_cost_prediction.logger import logging
import pandas as pd
import os, sys
import shutil

app = Flask(__name__,template_folder="templates")
 