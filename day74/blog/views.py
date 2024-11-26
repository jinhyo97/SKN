from django.shortcuts import render
from django.views.generic import ListView, DetailView
from .models import Post

# Create your views here.
# def index(reqeust):
#     posts = Post.objects.all().order_by('-pk')

#     return render(
#         reqeust,
#         "blog/index.html",
#         {
#             "posts": posts,
#         }
#     )


# def single_post_page(reqeust, pk):
#     post = Post.objects.get(pk=pk)

#     return render(
#         reqeust,
#         "blog/single_post_page.html",
#         {
#             "post": post,
#         }
#     )


class PostList(ListView):
    model = Post
    ordering = "-pk"


class PostDetail(DetailView):
    model = Post
