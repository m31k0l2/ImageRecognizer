class ImageNetEvolution(rateCount: Int=3): NetEvolution(0.2, rateCount) {
    override fun createNet() = CNetwork(1,2,2,2,20,10)
}