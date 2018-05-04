class ImageNetEvolution(rateCount: Int=3): NetEvolution(0.2, rateCount) {
    override fun createNet() = CNetwork(6, 6, 4, 4, 120, 10)
}