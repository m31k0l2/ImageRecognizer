fun main(args: Array<String>) {
    val net = ImageNet(40, 1, 0.01)
    net.evolute(10000, 40, 200)
}

