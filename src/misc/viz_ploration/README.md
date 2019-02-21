# Src: 

[James Powell Talk](https://www.youtube.com/watch?v=eEXKIp8h0T0)

# Goal:

Learn various technologies around DS and ML

- Flask
    - present results on a webpage

- Celery
    - Job server

- Bokeh

- Redis
    - Job queue for Celery


# Notes:

## Flask

### Problems:

1) Cannot serve multiple users at once

- Does not serve multiple users at once.

- CAN use Gunicorn to address this, but will be blocking

- Solution: Use a job queue which then queries if the job is done, and if it is, pulls the result
