from django.urls import path, include
from . import views


urlpatterns = [
    path("<int:pk>", views.PostDetail.as_view()),   # 127.0.0.1/blog/1
    path("", views.PostList.as_view())              # 127.0.0.1/blog/
]
