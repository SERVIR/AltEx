Web Application
======================================

**This app is created to run in the Teyths Platform programming environment.** 

**You can find a working demo here at** |http://tethys.servirglobal.net/apps/altex/|

.. |http://tethys.servirglobal.net/apps/altex/| raw:: html


    <a href="http://tethys.servirglobal.net/apps/altex/" target="_blank">http://tethys.servirglobal.net/apps/altex/ </a>

.. note::

    The following instructions have been tested on Ubuntu 16.04. Your workflow might be slightly different based on the operating system that you are using.


Prerequisites
--------------

-  Tethys Platform 2.0 (CKAN, PostgresQL, GeoServer): See:
   http://docs.tethysplatform.org
-  Geoserver needs CORS enabled.
-  SciPy (Python Package)
-  netCDF4 (Python Package)
-  NumPy (Python Package)
-  Git Large File Storage. |https://help.github.com/articles/installing-git-large-file-storage/#platform-linux|

.. |https://help.github.com/articles/installing-git-large-file-storage/#platform-linux| raw:: html

    <a href="https://help.github.com/articles/installing-git-large-file-storage/#platform-linux" target="_blank">Instructions here</a>

-  JASON2 Data from |ftp://avisoftp.cnes.fr/AVISO/pub/jason-2/gdr_d|

.. |ftp://avisoftp.cnes.fr/AVISO/pub/jason-2/gdr_d| raw:: html

    <a href="ftp://avisoftp.cnes.fr/AVISO/pub/jason-2/gdr_d" target="_blank">ftp://avisoftp.cnes.fr/AVISO/pub/jason-2/gdr_d</a>

Install SciPy
~~~~~~~~~~~~~~~~~~

Note: Before installing Psycopg2 into your python site-packages, activate
your Tethys conda environment using the alias `t`:

::

    $ t

::

    (tethys)$ conda install -c conda-forge scipy


Install netCDF4
~~~~~~~~~~~~~~~~~~

Note: Before installing Psycopg2 into your python site-packages, activate
your Tethys conda environment using the alias `t`:

::

    $ t

::

    (tethys)$ conda install -c conda-forge netCDF4


Web App Installation
----------------------

.. warning::

    Be sure to install Git Large file transfer. This step is critical to download the Geoid Mat file from the git repository.


Installation for App Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Clone the app into a directory of your choice

::

    $ t
    (tethys)$ git clone https://github.com/SERVIR/AltEx.git
    (tethys)$ cd AltEx
    (tethys)$ python setup.py develop


Open the :file:`jason.py` for editing using ``vim`` or any text editor of your choice:

::

    $ t
    (tethys)$ cd AltEx/tethysapp/altex
    (tethys)$ sudo vi jason.py

Press :kbd:`i` to start editing and enter the path to the JASON2 data directory. You can find it right after the import statements.

::

    # Replace this with the path to the data directory
    JASON_DIR = '/home/dev/avisoftp.cnes.fr/AVISO/pub/jason-2/gdr_d'

Press :kbd:`ESC` to exit ``INSERT`` mode and then press ``:x`` and :kbd:`ENTER` to save changes and exit.

Start the Tethys Server

::

    (tethys)$ tms


You should now have the AltEx app running on a development server on your machine. Tethys Platform provides a web interface called the Tethys Portal. You can access the app through the Tethys portal by opening http://localhost:8000/ (or if you provided custom host and port options to the install script then it will be <HOST>:<PORT>) in a new tab in your web browser.

Installation for Production
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Installing apps in a Tethys Platform configured for production can be challenging. Most of the difficulties arise, because Tethys is served by Nginx in production and all the files need to be owned by the Nginx user. The following instructions will allow you to deploy the AltEx web app on to your own Tethys production server. You can find the Tethys Production installation instructions `here. <http://docs.tethysplatform.org/en/stable/installation/production.html>`_


Change the Ownership of the Files to the Current User


*During the production installation any Tethys related files were change to be owned by the Nginx user. To make any changes on the server it is easiest to change the ownership back to the current user. This is easily done with an alias that was created in the tethys environment during the production installation process*


::

    $ t
    (tethys)$ tethys_user_own

Download App Source Code from GitHub

.. warning::

    Be sure to install Git Large file transfer. This step is critical to download the Geoid Mat file from the git repository.


::

    $ cd $TETHYS_HOME/apps/
    $ git clone https://github.com/SERVIR/AltEx

.. tip::

    Substitute $TETHYS_HOME with the path to the tethys main directory.

Open the :file:`jason.py` for editing using ``vim`` or any text editor of your choice:

::

    $ t
    (tethys)$ cd $TETHYS_HOME/apps/AltEx/tethysapp/altex
    (tethys)$ sudo vi jason.py


Press :kbd:`i` to start editing and enter the path to the JASON2 data directory. You can find it right after the import statements.

::

    # Replace this with the path to the data directory
    JASON_DIR = '/home/prod/jason-2/'

Press :kbd:`ESC` to exit ``INSERT`` mode and then press ``:x`` and :kbd:`ENTER` to save changes and exit.


Return to the main directory of the app. Then, execute the setup script (:file:`setup.py`) with the ``install`` command to make Python aware of the app and install any of its dependencies

::

    (tethys)$ cd $TETHYS_HOME/apps/AltEx/
    (tethys)$ python setup.py install

Collect Static Files and Workspaces

The static files and files in app workspaces are hosted by Nginx, which necessitates collecting all of the static files to a single directory and all workspaces to another single directory. These directory is configured through the ``STATIC_ROOT`` and ``TETHYS_WORKSPACES_ROOT`` setting in the :file:`settings.py` file. Collect the static files and workspaces with this command

::

    (tethys)$ tethys manage collectall

Change the Ownership of Files to the Nginx User

The Nginx user must own any files that Nginx is serving. This includes the source files, static files, and any workspaces that your app may have. The following alias will accomplish the change in ownership that is required

::

    (tethys)$ tethys_server_own
     

Restart uWSGI and Nginx services to effect the changes

::

    $ sudo systemctl restart tethys.uwsgi.service
    $ sudo systemctl restart nginx

.. note::

    For updating the app on production server, simply pull the app from GitHub. Once you have made a pull request (at times you may have to stash your local changes), follow the above steps to reinstall/update the app. You will have reenter the path to the jason2 data directory in the :file:`jason.py` file.


































