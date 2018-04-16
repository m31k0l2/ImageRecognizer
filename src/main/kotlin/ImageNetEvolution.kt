class ImageNetEvolution(rateCount: Int=3): NetEvolution(0.2, rateCount) {
    override fun createNet() = CNetwork(4, 2, 2, 2, 10)
}