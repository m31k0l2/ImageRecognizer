class ImageNetEvolution(rateCount: Int=3): NetEvolution(0.2, rateCount) {
    override fun createNet() = CNetwork(8, 6, 4, 10)
}