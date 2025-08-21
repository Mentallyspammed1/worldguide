 **Spaceship Prompt**
    *   **What it is:** A highly customizable Zsh prompt that aims to be feature-rich and modular.
    *   **Pros:** Packed with features (Git, Node, Python, Ruby, Docker, etc.), very configurable, supports Nerd Fonts.
    *   **Cons:** Can be slower than P10k or Pure due to its extensive checks, configuration can be complex.
    *   **Best for:** Zsh users who want a feature-rich prompt similar to P10k but with a different configuration style or module set."
git clone https://github.com/spaceship-prompt/spaceship-prompt.git ~/.spaceship-prompt && echo 'source ~/.spaceship-prompt/spaceship.zsh' >> ~/.zshrc && echo 'prompt spaceship' >> ~/.zshrc
