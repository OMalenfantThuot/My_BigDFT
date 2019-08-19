#!/usr/bin/env python
r"""
This script creates a file containing the documentation of all the
built-in attributes of the Logfile class. They are short-cuts to
get most of the relevant data by using a simple attribute getter.
"""
if __name__ == "__main__":

    from mybigdft.iofiles.logfiles import ATTRIBUTES

    msg = "These attributes allow to get most of the relevant data contained "\
          "in a Logfile. They provide simple shortcuts instead of having to "\
          "know exactly where to look for them in the yaml output file, "\
          "represented by the Logfile class. If some attributes do not "\
          "appear in the BigDFT output, they default to `None`."

    with open("source/attributes.rst", "w") as stream:
        stream.write(msg+"\n")
        for attr, description in ATTRIBUTES.items():
            stream.write("\n"+attr+"\n")
            stream.write("   :Returns: {}".format(description["doc"]+".\n"))
