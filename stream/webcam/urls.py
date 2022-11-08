from django.contrib import admin
from django.urls import path


from webcam import views
urlpatterns = [
    path('', views.index, name='index'),
    path('drone1/', views.video_feed_drone1, name='drone1'),
    path('drone2/', views.video_feed_drone2, name='drone2'),
    path('drone3/', views.video_feed_drone3, name='drone3'),
    path('camera1/', views.video_feed_camera1, name='camera1')
]
