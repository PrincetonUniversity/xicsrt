# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir pablant <npablant@pppl.gov>

A set of tools to help with automatic API documentation of XICSRT.

Description
-----------
XICSRT uses sphinx for documentation, and API docs are based on the idea of
code self documentation though `python` doc strings. This module contains
a set of decorators and helper function to aid in self documentation.

The most important part of this module is the `@dochelper` decorator which
should be used for all element classes.

.. Note:
    Philosophy: Python help() should be just as readable as the Sphinx webpage.

Todo:
    - The config docstrings should all be indented follow the help() standard.
    - Would it be helpful to show which inherited class the options came from?
"""

import inspect
import re


def dochelper(cls):
    """
    A functional wrapper for the DocHelper class. Intended to be used
    as a decorator.

    This decorator does the following:

    1. Adds a 'Configuration Options' section to the class docstring that
       contains all options defined in default_config and any class ancestors.

    """
    return DocHelper(cls)()

class DocHelper:
    """
    A class to help generate docstrings for XICSRT.

    This is expected to be used through the `@dochelper` class decorator.
    """

    def __init__(self, cls):
        self.cls = cls
        self.update_class_docstring(cls)

    def __call__(self):
        return self.cls

    def update_class_docstring(self, cls):
        """
        Update the class docstring. This method does the following:
        1. Creates a new section 'Configuration Options' which contains
           combined documentation for all config options defined in any
           ancestor.
        """
        flag_sort_alpha = False

        if cls.__doc__ == None:
            cls.__doc__ = ''

        sections = {}
        mro = list(inspect.getmro(cls))
        # The order here is important. As written, this will always keep
        # the section from the outermost class, while also adding sections
        # from outermost to innermost.
        for ancestor in mro:
            if hasattr(ancestor, 'default_config'):
                doc = ancestor.default_config.__doc__
                if doc is not None:
                    doc = inspect.getdoc(ancestor.default_config)
                    new = self._parse_config_doc(doc)

                    # I can't use update() because I don't want to overwrite values.
                    for key in new:
                        if key not in sections:
                            sections[key] = new[key]

        # Build up the new config docs from the section dictionary
        new_doc = ''
        key_list = list(sections.keys())
        if flag_sort_alpha:
            key_list.sort()
        for key in key_list:
            new_doc += sections[key]['name']
            if sections[key]['sig']:
                new_doc += ' : ' + sections[key]['sig']
            new_doc += '\n'
            new_doc += sections[key]['body']
            new_doc += '\n'

        cls.__doc__ += '\n'
        cls.__doc__ += 'Configuration Options:\n\n'
        cls.__doc__ += new_doc

    def _parse_config_doc(self, doc):
        """
        Parse configuration documentation into sections. This assumes that
        the config docs are well formatted.
        """
        section_dict = {}
        name = None
        lines = doc.splitlines()
        for line in lines:
            m = re.match(r'^([\w]+)\s*:?\s*(.*)', line)
            if m:
                name = m.group(1)
                section_dict[name] = {}
                section_dict[name]['name'] = m.group(1)
                section_dict[name]['sig'] = m.group(2)
                section_dict[name]['body'] = ''
            else:
                if name:
                    if line.strip():
                        section_dict[name]['body'] += line + '\n'

        return section_dict
