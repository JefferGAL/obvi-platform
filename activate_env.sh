#!/bin/bash
echo "Activating LlamaIndex environment..."
source venv_llamaindex/bin/activate
echo "Environment activated! Python path:"
which python
echo "Ready for LlamaIndex development!"
exec "$SHELL"
