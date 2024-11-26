from django.shortcuts import render
from django.views.generic import ListView, DetailView
from .models import Post

# Create your views here.
# def index(reqeust):
#     posts = Post.objects.all()

#     return render(
#         reqeust,
#         "news/index.html",
#         {
#             "posts": posts,
#         }
#     )


class PostList(ListView):
    model = Post
    ordering = "-pk"


class PostDetail(DetailView):
    model = Post
