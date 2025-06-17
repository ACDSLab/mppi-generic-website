---
title: Blog Posts
layout: page
description: ""
---
## Ordered By Date
<ul>
  {% for post in site.posts %}
    <li>
    <big><big><a href="{{ post.url }}">{{ post.title }}</a> -
    <time datetime="{{ post.date | date: "%Y-%m-%d"}}">{{ post.date | date_to_long_string }}</time>:</big></big>
    {{ post.description }}
    </li>
  {% endfor %}
</ul>

