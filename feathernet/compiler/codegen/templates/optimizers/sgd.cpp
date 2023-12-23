class SGDOptimizer {
public:
    SGDOptimizer(float lr) : learning_rate(lr) {}

    void update(std::vector<float>& weights, const std::vector<float>& grad) {
        assert(weights.size() == grad.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learning_rate * grad[i];
        }
    }
}

private:
    float learning_rate;
};
