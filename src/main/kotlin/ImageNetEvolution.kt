class ImageNetEvolution(rateCount: Int=3): NetEvolution(0.2, rateCount) {
    override fun createNet() = CNetwork(4,5,2,3,40,10)
}