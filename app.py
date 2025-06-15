from flask import Flask, request, jsonify, render_template
from fetch_comments import fetch_comments
from text_processing import TextProcessor
from analyze import predict_comments
import joblib
import numpy as np
from collections import Counter
import re