from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('',views.home,name="home"),
    path('user_register/',views.user_register,name="user_register"),
    path('user_login/',views.user_login,name="user_login"),
    path('auth_login/',views.auth_login,name='auth_login'),
    path('user_dashboard/', views.user_dashboard, name='user_dashboard'),
    path('auth_dashboard/',views.auth_dashboard,name="auth_dashboard"),
    path('user_logout/', views.user_logout, name='user_logout'),
    path('auth_logout',views.auth_logout,name='auth_logout'),
    path('chatbot/', views.chatbot_api, name='chatbot_api'),
    path("save-location/", views.save_location, name="save_location"),
    path("get-locations/", views.get_locations, name="get_locations"),
    path("delete-location/", views.delete_location, name="delete_location"),
    path('camera/live/05/', views.live_camera_feed, name='camera05'),
    path("submit-report/", views.submit_garbage_report, name="submit_garbage_report"),
    path("authority/user-reports/", views.user_reports,name='user_reports'),
    path("authority/user-reports/<int:report_id>/download/",views.download_report_zip,name="download_report_zip"),




    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
