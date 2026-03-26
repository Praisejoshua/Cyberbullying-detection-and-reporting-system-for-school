# detection/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze-text/', views.analyze_text, name='analyze_text'),
    path('analyze-image/', views.analyze_image, name='analyze_image'),
    path('api/analyze/', views.api_analyze, name='api_analyze'),
]