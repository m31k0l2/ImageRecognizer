class ImageNetEvolution(rateCount: Int=3): NetEvolution(0.2, rateCount) {
//    override fun createNet() = CNetwork(2, 4, 10, 10) /v0123
    override fun createNet() = CNetwork(4, 4, 4, 4, 20, 10)
}