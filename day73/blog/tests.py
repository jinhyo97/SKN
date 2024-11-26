from django.test import TestCase, Client
from django.contrib.auth.models import User

from bs4 import BeautifulSoup

from .models import Post, Category


# Create your tests here.

class TestView(TestCase):
    def setUp(self):
        self.client = Client()

        self.user_trump = User.objects.create_user(
            username='trump',
            password='password',
        )
        self.user_obama = User.objects.create_user(
            username='obama',
            password='password',
        )

        self.category_society = Category.objects.create(
            name='society',
            slug='society'
        )
        self.category_culture = Category.objects.create(
            name='culture',
            slug='culture'
        )

    def test_post_list(self):
        response = self.client.get('/blog/')
        self.assertEqual(response.status_code, 200)

        bs = BeautifulSoup(response.content, 'lxml')
        self.assertEqual(bs.title.text, 'Blog')

        navbar = bs.navbar
        self.assertIn('Blog', navbar.text)
        self.assertIn('About Me', navbar.text)

        self.assertEqual(Post.objects.count(), 3)

    def test_post_detail(self):
        post_001 = Post.objects.create(
            title="첫번째 포스트",
            content="Hello World!"
        )

        self.assertEqual(post_001.get_absolute_url(), '/blog/1')
        response = self.client.get(post_001.get_absolute_url())
        self.assertEqual(response.status_code, 200)
        bs = BeautifulSoup(response.content, 'lxml')

        navbar = bs.nav
        self.assertIn('Blog', navbar.text)
        self.assertIn('About me', navbar.text)

        self.assertIn(post_001.title, bs.title.text)

        main_area = bs.select_one('div#main-area')
        post_area = bs.select_one('div#post-area')
        self.assertIn(post_001.title, post_area.text)
        self.assertIn(post_001.content, post_area.text)