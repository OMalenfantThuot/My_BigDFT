Logfile
-------

The :class:`~mybigdft.iofiles.Logfile` class is the base class you want to be
using all the time. It is mainly meant to manipulate the output of a BigDFT
calculation (written using the YAML format).

However, it might happen that the output actually contains many documents (for
instance, one document per geometry optimization procedure). In such cases,
the initialization of a ``Logfile`` actually gives another type of object,
deriving from the :class:`~mybigdft.iofiles.MultipleLogfile` class. Be careful:
these objects behave as a list of ``Logfile`` instances, not as a ``Logfile``
instance (even though they are initialized *via* the ``Logfile`` class).
To keep the same example as above, the output file of a geometry optimization
calculation can be read via the :meth:`~mybigdft.iofiles.Logfile.from_file`
method of the ``Logfile`` class, returning a
:class:`~mybigdft.iofiles.GeoptLogfile` instance.

.. data:: mybigdft.iofiles.ATTRIBUTES
    :annotation: Definition of the base attributes of a Logfile instance (and
        where to look for them in the logfile).

.. autoclass:: mybigdft.iofiles.Logfile
    :special-members: __getattr__, __setattr__, __dir__ 

.. autoclass:: mybigdft.iofiles.MultipleLogfile

.. autoclass:: mybigdft.iofiles.GeoptLogfile
