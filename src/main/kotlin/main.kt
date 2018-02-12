fun main(args: Array<String>) {
    val net = ImageNet(30, 1, 0.005)
    net.evolute(10000, 20, 500)
}

