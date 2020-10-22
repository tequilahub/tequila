:orphan:
{% if referencefile %}
.. include:: {{ referencefile }}
{% endif %}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}


.. autoclass:: {{fullname}}
   :undoc-members:
   :show-inheritance:
   :special-members:

   .. automethod:: __init__
      :noindex:

   {% block attributes_summary %}
   {% if attributes %}

   .. rubric:: Attributes

   .. autosummary::
      :toctree: {{ objname }}
      :nosignatures:
   {% for item in attributes %}
      {%- if not item.startswith('_') or not item in ['__init__'] %}
      ~{{ fullname }}.{{ item }}
      {%- endif -%}
   {%- endfor %}

   {% endif %}
   {% endblock %}


   {% block methods_summary %}
   {% if methods %}

   .. rubric:: Methods

   .. autosummary::
      :toctree: {{ objname }}
      :nosignatures:
   {% for item in methods %}
      {%- if not item.startswith('_') or not item in ['__init__'] %}
      ~{{ fullname }}.{{ item }}
      {%- endif -%}
   {%- endfor %}

   {% endif %}
   {% endblock %}


