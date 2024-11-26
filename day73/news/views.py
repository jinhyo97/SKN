from django.shortcuts import render
from django.views.generic import ListView
from .models import Post


# Create your views here.
# def index(request):
#     posts = Post.objects.all()

#     return render(
#         request,
#         "news/index.html",
#         {
#             "posts" : posts,
#         },
#     )


def single_post_page(request, pk):
    post = Post.objects.get(pk=pk)

    return render(
        request,
        "blog/sigle_post_page.html",
        {
            "post" : post,
        },
    )   


class PostList(ListView):
    model = Post
    # template_name = 'news/index.html'