import os
import sys
import time
from datetime import date
import shutil

template =  """Title: {0}
Date: {1}
Category: Python
Tags: {2}
Slug: {3}
Author: Anisha Keshavan
Summary: {4}

{6}% notebook {5} %{7}

"""


if __name__ == "__main__":
    if len(sys.argv)==2:
        nb = sys.argv[1]
        nbname = os.path.split(nb)[1]
        nbpath = os.path.join("notebooks",nbname)
        if os.path.exists(nbpath):
            print "updating notebook %s"%nb
        else:
            print "adding notebook %s"%nb
            name = raw_input("Name of notebook:")
            tags = raw_input("Tags:")
            slug = name.replace(" ","-")
            today = date.today()
            time = today.strftime("%A %d. %B %Y")
            summary = raw_input("Summary:")
            T = template.format(name,date,tags,slug,summary,nbpath,"{","}")
            cname = os.path.join("content",nbname.replace(".ipynb",".md"))
            with open(cname,"w") as f:
                f.write(T)
            os.system("git add %s"%cname)
        shutil.copy(nb,nbpath)
        os.system("git add %s"%nbpath)
        os.system('git commit -m "notebook %s updated"%nbname')
