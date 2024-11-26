from django.urls import path, include
from . import views


# FBV, CBV
urlpatterns = [
    # path("", views.index),
    # path("<int:pk>/", views.single_post_page),
    path("", views.PostList.as_view()),
    path("<int:pk>/", views.PostDetail.as_view()),
]
